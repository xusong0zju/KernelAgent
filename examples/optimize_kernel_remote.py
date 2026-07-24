#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""通用算子优化驱动器：本机 LLM 生成 + 远程 GPU daemon 真测，迭代优化。

复用 RMSNorm 闭环（见 docs/RMSNorm优化实录.md），把"算子包"参数化：
每个算子包 = 一个目录含 problem.py(input)/input.py(初始kernel)/test.py +
一份 operator.yaml 描述（算子说明、优化方向、目标 GPU）。

用法：
  env -u ALL_PROXY -u all_proxy OPENAI_MODEL=deepseek-v4-pro \
    python examples/optimize_kernel_remote.py \
      --kernel-dir examples/optimize_voxelization \
      --url http://127.0.0.1:8765 --token <daemon-token> --rounds 3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests

os.environ.pop("ANTHROPIC_API_KEY", None)
HERE = Path(__file__).resolve().parent

# ---------- LLM ----------

def _extract_code(text: str) -> str | None:
    # Prefer a properly-closed ```python ... ``` block.
    m = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL) or re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        for b in m:
            if re.search(r"\bdef\s+kernel_function\b", b):
                return b.strip()
        return max(m, key=len).strip()
    # Fallback: an OPEN code block (reasoning models sometimes get cut off
    # mid-block). Take from ```python to end-of-string.
    om = re.search(r"```python\s*\n(.*)$", text, re.DOTALL)
    if om:
        block = om.group(1)
        if re.search(r"\bdef\s+kernel_function\b", block):
            return block.strip()
    return None


def _call_llm(prompt: str, model: str, attempts: int = 8) -> str:
    """Call the LLM, retrying if no code block is produced.

    deepseek-v4-pro is a reasoning model that frequently spends its entire
    token budget on the thinking block and never emits a final text/code
    answer. Output is stochastic, so retrying usually lands a clean answer
    within a handful of tries.
    """
    from utils.providers import get_model_provider
    provider = get_model_provider(model)
    temps = [0.3, 0.5, 0.2, 0.4]
    last = ""
    for i in range(attempts):
        try:
            resp = provider.get_response(
                model, [{"role": "user", "content": prompt}],
                max_tokens=16000, temperature=temps[i % len(temps)],
            )
            last = resp.content
            if _extract_code(last) is not None:
                return last
            print(f"  [llm] attempt {i+1}: no usable code block (len={len(last)})")
        except Exception as e:  # noqa: BLE001
            print(f"  [llm] attempt {i+1} error: {e}")
            continue
    return last


def _extract_code(text: str) -> str | None:
    m = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL) or re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return None
    for b in m:
        if re.search(r"\bdef\s+kernel_function\b", b):
            return b.strip()
    return max(m, key=len).strip()


# ---------- daemon ----------

def _post(url, token, path, item):
    r = requests.post(f"{url}/{path}", json={"items": [item], "token": token},
                      headers={"Authorization": f"Bearer {token}"}, timeout=300)
    r.raise_for_status()
    return r.json()["results"][0]


def verify(url, token, kernel_code, problem_code, test_code):
    res = _post(url, token, "run_test_batch", {"kernel_code": kernel_code, "problem_code": problem_code, "test_code": test_code})
    return bool(res.get("success")), res.get("stdout", ""), res.get("stderr", "")


def benchmark(url, token, kernel_code, problem_code, warmup=10, repeat=50):
    res = _post(url, token, "benchmark_batch", {"kernel_code": kernel_code, "problem_code": problem_code, "warmup": warmup, "repeat": repeat, "kind": "kernel"})
    t = res.get("time_ms")
    return float(t) if t is not None else float("inf")


def benchmark_eager(url, token, problem_code, warmup=10, repeat=50):
    res = _post(url, token, "benchmark_batch", {"problem_code": problem_code, "warmup": warmup, "repeat": repeat, "kind": "eager"})
    t = res.get("time_ms")
    return float(t) if t is not None else float("inf")


# ---------- prompt ----------

OPT_PROMPT = """\
Rewrite this Triton kernel to be faster. Keep correctness (atol/rtol=1e-2).

Target: {gpu}
Baseline: eager {eager_ms:.3f} ms | current {current_ms:.3f} ms | best {best_ms:.3f} ms
Problem: {problem_short}
Directions:
{opt_directions}

Triton constraints (avoid these — they crash compile):
- no 2D tensor column indexing (t[:, i] unsupported); use 1D loads
- no indexing a vector with another vector element (lin[i] where i is
  constexpr) — compute directly, broadcast via tl.arange
- use tl.maximum/tl.minimum, not .clamp
- constexpr for anything passed to tl.arange/tl.static_range
- NO `continue`/`break` inside @triton.jit (unsupported AST); use masks only
- NO tl.static_shared / tl.shared / _semantic (do not exist); use registers + tl.atomic_add
- no torch.nn / torch.matmul / torch.ops.aten.* compute inside kernels

Current kernel:
```python
{current_kernel}
```

Output ONLY one ```python block with the complete optimized kernel file
(imports + @triton.jit + kernel_function). No prose, no analysis before the block.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", required=True, help="dir with problem.py/input.py/test.py + operator.yaml")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--token", default=os.environ.get("KA_DAEMON_TOKEN", ""))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    kd = Path(args.kernel_dir).resolve()
    problem_code = (kd / "problem.py").read_text()
    test_code = (kd / "test.py").read_text()
    initial = (kd / "input.py").read_text()
    op_yaml = _load_op_yaml(kd)

    print("=" * 72)
    print(f"Optimizing: {op_yaml.get('name', kd.name)}")
    print(f"  target GPU: {op_yaml.get('gpu','NVIDIA RTX 2080 Ti (Turing SM7.5, 68 SMs, 616 GB/s)')}")
    print("=" * 72)

    eager_ms = benchmark_eager(args.url, args.token, problem_code, args.warmup, args.repeat)
    print(f"[baseline] eager = {eager_ms:.3f} ms")
    ok, _, err = verify(args.url, args.token, initial, problem_code, test_code)
    if not ok:
        print(f"  initial kernel FAILED verify:\n{err[:400]}"); sys.exit(1)
    cur_ms = benchmark(args.url, args.token, initial, problem_code, args.warmup, args.repeat)
    print(f"[baseline] initial kernel = {cur_ms:.3f} ms  (eager {eager_ms:.3f}, kernel {cur_ms/eager_ms*100:.0f}% of eager)")

    best_kernel, best_ms = initial, cur_ms
    best_src = "triton-initial"

    # Optional CUDA candidate: a hand-written kernel_cuda.py in the dir is
    # treated as a baseline candidate alongside the Triton ones. The daemon
    # runs it via the same run_test/benchmark path (kernel_cuda uses
    # torch.utils.cpp_extension; the compiled .so is cached on the box).
    cuda_file = kd / "kernel_cuda.py"
    cuda_code = cuda_file.read_text() if cuda_file.exists() else None
    if cuda_code:
        print("[cuda] verifying hand-written CUDA candidate ...", flush=True)
        ok, _, cerr = verify(args.url, args.token, cuda_code, problem_code, test_code)
        if ok:
            cuda_ms = benchmark(args.url, args.token, cuda_code, problem_code, args.warmup, args.repeat)
            print(f"[cuda] CUDA candidate = {cuda_ms:.3f} ms  (vs triton-initial {cur_ms:.3f})")
            if cuda_ms < best_ms:
                best_ms, best_kernel, best_src = cuda_ms, cuda_code, "cuda"
                print(f"[cuda] → CUDA is now best ({cuda_ms:.3f} ms)")
        else:
            print(f"[cuda] CUDA candidate FAILED verify: {cerr[-160:]}")

    print(f"[goal] beat {best_ms:.3f} ms (current best: {best_src})\n")

    for r in range(1, args.rounds + 1):
        print(f"--- round {r}/{args.rounds} ---")
        prompt = OPT_PROMPT.format(
            gpu=op_yaml.get("gpu", "NVIDIA RTX 2080 Ti (Turing SM7.5, 68 SMs, 616 GB/s)"),
            problem_short=op_yaml.get("problem_short", op_yaml.get("problem", "see problem.py")),
            current_kernel=best_kernel, eager_ms=eager_ms,
            current_ms=best_ms, best_ms=best_ms,
            opt_directions=op_yaml.get("directions", ""),
        )
        print(f"  [llm] {args.model} generating ...", flush=True)
        cand = _extract_code(_call_llm(prompt, args.model))
        if cand is None:
            print("  [llm] no code block; skipping"); continue
        print("  [verify] remote GPU ...", flush=True)
        ok, stdout, stderr = verify(args.url, args.token, cand, problem_code, test_code)
        if not ok:
            print(f"  [verify] FAILED (keep best). {stderr[-200:].strip()}"); continue
        cand_ms = benchmark(args.url, args.token, cand, problem_code, args.warmup, args.repeat)
        sp = (best_ms - cand_ms) / best_ms * 100
        print(f"  [bench] {cand_ms:.3f} ms vs best {best_ms:.3f} ms ({sp:+.1f}%)  {'✅ FASTER' if cand_ms < best_ms else '⏸ not faster'}")
        if cand_ms < best_ms:
            best_ms, best_kernel, best_src = cand_ms, cand, f"triton-round{r}"
            (kd / f"best_round{r}.py").write_text(best_kernel)
            print(f"  → saved best_round{r}.py")

    print("\n" + "=" * 72 + "\nRESULT")
    print(f"  initial : {cur_ms:.3f} ms")
    print(f"  best    : {best_ms:.3f} ms   ({(1-best_ms/cur_ms)*100:+.1f}% vs initial)  [{best_src}]")
    print(f"  eager   : {eager_ms:.3f} ms   (best is {best_ms/eager_ms*100:.0f}% of eager)")
    (kd / "best.py").write_text(best_kernel)
    print(f"  saved   : {kd/'best.py'}")
    print("=" * 72)


def _load_op_yaml(kd: Path) -> dict:
    """Tiny YAML-ish loader for operator.yaml (name/gpu/problem/directions)."""
    p = kd / "operator.yaml"
    if not p.exists():
        return {"name": kd.name, "problem": "see problem.py", "directions": "see problem.py"}
    d = {}
    cur = None
    for line in p.read_text().splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line:
            k, _, v = line.partition(":")
            v = v.strip()
            if v:
                d[k.strip()] = v
            else:
                cur = k.strip(); d[cur] = ""
        elif cur:
            d[cur] += line.strip() + "\n"
    return d


if __name__ == "__main__":
    main()
