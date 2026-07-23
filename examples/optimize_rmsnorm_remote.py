#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iteratively optimize the RMSNorm kernel (Qwen-style normalization)
using KernelAgent's pieces: local LLM (deepseek-v4-pro via KingCloud) to
generate optimized kernels + the remote GPU daemon to verify correctness
and benchmark time on an RTX 2080 Ti.

This is a benchmark-driven optimization loop (no NCU bottleneck analysis),
suitable when the GPU isn't in the specs table or NCU matching is flaky.
The optimization direction is supplied as a prior (memory-bound, two-pass
→ single-pass online reduction) — exactly the "evaluate optimization space"
exercise.

Usage:
  # 1. deploy the daemon first (see docs/远程GPU部署.md)
  # 2. ensure ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL env for the LLM
  env -u ALL_PROXY -u all_proxy OPENAI_MODEL=deepseek-v4-pro \
    ~/miniconda3/envs/kernel_agent/bin/python examples/optimize_rmsnorm_remote.py \
      --url http://127.0.0.1:8765 --token <daemon-token> --rounds 3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path

import requests

# Force the token-path LLM (KingCloud deepseek-v4-pro).
os.environ.pop("ANTHROPIC_API_KEY", None)

HERE = Path(__file__).resolve().parent
PROBLEM_FILE = HERE / "optimize_qwen_rmsnorm" / "problem.py"
INITIAL_FILE = HERE / "optimize_qwen_rmsnorm" / "input.py"
TEST_FILE = HERE / "optimize_qwen_rmsnorm" / "test.py"


# ---------------------------------------------------------------------------
# LLM (local agent brain)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, model: str) -> str:
    from utils.providers import get_model_provider

    provider = get_model_provider(model)
    resp = provider.get_response(model, [{"role": "user", "content": prompt}], max_tokens=12000, temperature=0.3)
    return resp.content


def _extract_code(text: str) -> str | None:
    m = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        m = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return None
    # Prefer the block defining kernel_function.
    for b in m:
        if re.search(r"\bdef\s+kernel_function\b", b):
            return b.strip()
    return max(m, key=len).strip()


# ---------------------------------------------------------------------------
# Remote GPU daemon (verify + benchmark)
# ---------------------------------------------------------------------------

def _post(url, token, path, item):
    r = requests.post(
        f"{url}/{path}", json={"items": [item], "token": token},
        headers={"Authorization": f"Bearer {token}"}, timeout=300,
    )
    r.raise_for_status()
    return r.json()["results"][0]


def verify(url, token, kernel_code, problem_code, test_code):
    res = _post(url, token, "run_test_batch",
                {"kernel_code": kernel_code, "problem_code": problem_code, "test_code": test_code})
    return bool(res.get("success")), res.get("stdout", ""), res.get("stderr", "")


def benchmark(url, token, kernel_code, problem_code, warmup=10, repeat=50):
    res = _post(url, token, "benchmark_batch",
                {"kernel_code": kernel_code, "problem_code": problem_code,
                 "warmup": warmup, "repeat": repeat, "kind": "kernel"})
    t = res.get("time_ms")
    return float(t) if t is not None else float("inf")


def benchmark_eager(url, token, problem_code, warmup=10, repeat=50):
    res = _post(url, token, "benchmark_batch",
                {"problem_code": problem_code, "warmup": warmup, "repeat": repeat, "kind": "eager"})
    t = res.get("time_ms")
    return float(t) if t is not None else float("inf")


# ---------------------------------------------------------------------------
# Optimization prompt
# ---------------------------------------------------------------------------

OPT_PROMPT = """\
You are a world-class Triton kernel engineer. Optimize the RMSNorm kernel below.

## Target GPU
NVIDIA RTX 2080 Ti (Turing, SM 7.5, 68 SMs, 616 GB/s HBM bandwidth, 11 GB).

## Problem
RMSNorm over the channel/feature dimension (dim=1) of an NCHW tensor
[112, 64, 512, 512], float32 (~7.6 GB). y = x / sqrt(mean(x^2, dim=1) + eps).

## Current kernel
```python
{current_kernel}
```

## Measured performance
- PyTorch eager baseline: {eager_ms:.3f} ms
- Current kernel: {current_ms:.3f} ms  (best so far: {best_ms:.3f} ms)
- Theoretical bandwidth lower bound (single pass, 7.6GB @ 616GB/s): ~12.4 ms

## Optimization directions to consider
1. The current kernel does TWO passes over the input (first to compute
   sum-of-squares, then to normalize). FUSE into a SINGLE pass using an
   online/running sum-of-squares — halves memory traffic, the dominant cost.
2. Coalesced loads/stores along the contiguous W dimension.
3. Choose BLOCK sizes to maximize occupancy on 68 SMs (Turing).
4. Avoid shared-memory bank conflicts; use tl.dot or tl.reduce carefully.
5. Keep numerical correctness: atol=1e-2, rtol=1e-2 (BF16-ish tolerance ok).

## Hard constraints (KernelAgent runtime rules)
- All compute inside the @triton.jit kernel. Wrapper only validates/allocates/launches.
- Main entry must be `def kernel_function(x, ...) -> torch.Tensor` (test imports it).
- No torch.nn / torch.nn.functional / torch.matmul / torch.ops.aten.* compute.

## Output
Output ONLY one ```python block with the complete optimized kernel file
(imports + @triton.jit kernel + kernel_function wrapper). No explanation outside the block.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--token", default=os.environ.get("KA_DAEMON_TOKEN", ""))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    problem_code = PROBLEM_FILE.read_text()
    test_code = TEST_FILE.read_text()
    initial = INITIAL_FILE.read_text()

    print("=" * 72)
    print("RMSNorm optimization (Qwen-style) via KernelAgent + remote RTX 2080 Ti")
    print("=" * 72)

    # 1. baselines
    print("\n[baseline] PyTorch eager ...", flush=True)
    eager_ms = benchmark_eager(args.url, args.token, problem_code, args.warmup, args.repeat)
    print(f"[baseline] eager = {eager_ms:.3f} ms")

    print("[baseline] verifying initial kernel ...", flush=True)
    ok, _, err = verify(args.url, args.token, initial, problem_code, test_code)
    if not ok:
        print(f"  initial kernel FAILED verify:\n{err[:400]}")
        sys.exit(1)
    cur_ms = benchmark(args.url, args.token, initial, problem_code, args.warmup, args.repeat)
    print(f"[baseline] initial kernel = {cur_ms:.3f} ms  (eager {eager_ms:.3f} ms, "
          f"kernel is {cur_ms/eager_ms*100:.0f}% of eager)")

    best_kernel = initial
    best_ms = cur_ms
    print(f"\n[goal] beat {best_ms:.3f} ms; bandwidth-bound theoretical floor ~12.4 ms "
          f"(~{(1-12.4/best_ms)*100:.0f}% headroom)\n")

    # 2. iterative optimization
    for r in range(1, args.rounds + 1):
        print(f"--- round {r}/{args.rounds} ---")
        prompt = OPT_PROMPT.format(
            current_kernel=best_kernel, eager_ms=eager_ms,
            current_ms=best_ms, best_ms=best_ms,
        )
        print(f"  [llm] {args.model} generating ...", flush=True)
        raw = _call_llm(prompt, args.model)
        cand = _extract_code(raw)
        if cand is None:
            print("  [llm] no code block extracted; skipping")
            continue
        # verify
        print("  [verify] running on remote GPU ...", flush=True)
        ok, stdout, stderr = verify(args.url, args.token, cand, problem_code, test_code)
        if not ok:
            print(f"  [verify] FAILED (keep current best). stderr: {stderr[-200:].strip()}")
            continue
        cand_ms = benchmark(args.url, args.token, cand, problem_code, args.warmup, args.repeat)
        speedup = (best_ms - cand_ms) / best_ms * 100
        tag = "✅ FASTER" if cand_ms < best_ms else "⏸ not faster"
        print(f"  [bench] {cand_ms:.3f} ms vs best {best_ms:.3f} ms ({speedup:+.1f}%)  {tag}")
        if cand_ms < best_ms:
            best_ms = cand_ms
            best_kernel = cand
            (HERE / f"rmsnorm_best_round{r}.py").write_text(best_kernel)
            print(f"  → new best saved: rmsnorm_best_round{r}.py")

    # 3. summary
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  initial : {cur_ms:.3f} ms")
    print(f"  best    : {best_ms:.3f} ms   ({(1-best_ms/cur_ms)*100:+.1f}% vs initial)")
    print(f"  eager   : {eager_ms:.3f} ms   (best is {best_ms/eager_ms*100:.0f}% of eager)")
    print(f"  bw floor: ~12.4 ms          (best is {12.4/best_ms*100:.0f}% of bandwidth limit)")
    out = HERE / "rmsnorm_best.py"
    out.write_text(best_kernel)
    print(f"  saved   : {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
