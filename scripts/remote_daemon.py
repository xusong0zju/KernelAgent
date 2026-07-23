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

"""Remote GPU daemon for KernelAgent (route 2).

A FastAPI service that runs on the GPU box, reached over an ssh tunnel
from the local agent brain. It exposes the three GPU-touching operations
the local ``Remote*`` platform components call:

  POST /run_test_batch    — run kernel+test, return correctness (stdout/stderr)
  POST /benchmark_batch   — CUDA-event time a kernel / PyTorch eager / compiled
  POST /profile_batch     — NCU profile a kernel, return metrics dict
  GET  /health            — liveness + GPU name
  GET  /specs             — GPU specs dict (for the local gpu_name config)

Each batch endpoint takes ``{"items": [...], "token": ...}`` and returns
``{"results": [...]}`` (one result per item, same order). Batches are
executed serially for now — correctness first; concurrent execution is
step 6 of the plan.

The daemon REUSES the existing modules unchanged:
  - worker_util._run_test_multiprocess  (test execution)
  - benchmarking.benchmark.Benchmark    (CUDA-event timing + PTX capture)
  - profiling.kernel_profiler.KernelProfiler (NCU)

so local and remote behaviour stay identical. It must run with a Python
that has torch + triton installed (the deployer installs the editable
KernelAgent package on the cloud box, which brings these deps).

Usage:
  python scripts/remote_daemon.py --port 8765 --token <secret>
  # token may also live in --token-file (one line) or KA_DAEMON_TOKEN env.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger("remote_daemon")

app = FastAPI(title="KernelAgent remote GPU daemon")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    import traceback as _tb

    logger.error("unhandled in %s: %s", request.url.path, exc)
    logger.error(_tb.format_exc())
    # Return a structured (non-500-HTML) body so the Remote* client can
    # parse it; use 500 status but JSON content.
    return JSONResponse(
        status_code=500,
        content={"results": [], "error": f"{type(exc).__name__}: {exc}"},
    )


# --- auth state, set in main() / startup -------------------------------
_TOKEN: str | None = None


def _check_token(authorization: str | None, body_token: str | None) -> None:
    """Validate bearer token from header OR body field (Remote* sends both)."""
    if _TOKEN is None:
        return  # no token configured → open (dev only); deployer always sets one
    presented = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization.split(None, 1)[1].strip()
    elif body_token:
        presented = body_token
    if presented != _TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


# --- helpers -----------------------------------------------------------


def _write_tmp(content: str, name: str, workdir: Path) -> Path:
    p = workdir / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_workdir(prefix: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d


def _gpu_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        logger.warning("gpu detect failed: %s", e)
        return None


def _json_safe_float(v: Any) -> Any:
    """Convert non-JSON float values (inf/-inf/nan) to None for transport.

    JSON has no representation for inf/nan; the Remote* client restores
    None back to ``float("inf")`` semantics (see remote.py)."""
    import math

    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# --- endpoints ---------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "gpu": {"name": _gpu_name()}}


@app.get("/specs")
def specs(device_name: str | None = None) -> dict[str, Any]:
    """Return GPU specs. ``device_name`` optional; auto-detect if absent."""
    try:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import get_gpu_specs

        name = device_name or _gpu_name()
        if not name:
            return {"error": "no GPU detected and no device_name given"}
        return get_gpu_specs(name)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@app.post("/run_test_batch")
async def run_test_batch(request: Request, authorization: str | None = Header(None)) -> JSONResponse:
    body = await request.json()
    _check_token(authorization, body.get("token"))
    items = body.get("items", [])

    # Import lazily so the daemon can still /health without torch.
    from triton_kernel_agent.worker_util import _run_test_multiprocess

    results: list[dict[str, Any]] = []
    for item in items:
        try:
            workdir = _make_workdir("rt_")
            kernel_code = item["kernel_code"]
            problem_code = item["problem_code"]
            test_code = item["test_code"]
            test_list = test_code if isinstance(test_code, list) else [test_code]

            _write_tmp(problem_code, "problem.py", workdir)
            kernel_file = _write_tmp(kernel_code, "kernel.py", workdir)
            # Write each test as test_N.py; _run_test_multiprocess runs them in order.
            test_files = [
                _write_tmp(tc, f"test_{i}.py", workdir) for i, tc in enumerate(test_list)
            ]

            success, stdout, stderr = _run_test_multiprocess(
                logger=logger, workdir=workdir, test_files=test_files
            )
            results.append({"success": success, "stdout": stdout, "stderr": stderr})
        except Exception as e:  # noqa: BLE001
            results.append(
                {"success": False, "stdout": "", "stderr": f"{e}\n{traceback.format_exc()}"}
            )
    return JSONResponse({"results": results})


@app.post("/benchmark_batch")
async def benchmark_batch(request: Request, authorization: str | None = Header(None)) -> JSONResponse:
    body = await request.json()
    _check_token(authorization, body.get("token"))
    items = body.get("items", [])

    from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import Benchmark

    import threading

    results: list[dict[str, Any]] = []
    for item in items:
        try:
            workdir = _make_workdir("bm_")
            warmup = int(item.get("warmup", 25))
            repeat = int(item.get("repeat", 100))
            kind = item.get("kind", "kernel")
            problem_file = _write_tmp(item["problem_code"], "problem.py", workdir)

            # Benchmark requires a lock object (BenchmarkLockManager calls
            # lock.acquire()). The daemon is single-process, so a plain
            # threading.Lock serializes the (already serial) batch safely.
            bench = Benchmark(
                logger=logger,
                artifacts_dir=workdir,
                benchmark_lock=threading.Lock(),
                worker_id=-1,
                warmup=warmup,
                repeat=repeat,
            )
            if kind == "kernel":
                kernel_file = _write_tmp(item["kernel_code"], "kernel.py", workdir)
                r = bench.benchmark_kernel(kernel_file, problem_file)
            elif kind == "eager":
                r = bench.benchmark_pytorch(problem_file)
            elif kind == "compiled":
                r = bench.benchmark_pytorch_compile(problem_file)
            else:
                raise ValueError(f"unknown benchmark kind: {kind}")
            results.append(
                {
                    "time_ms": _json_safe_float(r.get("time_ms", float("inf"))),
                    "ptx_hash": r.get("ptx_hash"),
                    "error": None,
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append({"time_ms": None, "ptx_hash": None, "error": str(e)})
    return JSONResponse({"results": results})


@app.post("/profile_batch")
async def profile_batch(request: Request, authorization: str | None = Header(None)) -> JSONResponse:
    body = await request.json()
    _check_token(authorization, body.get("token"))
    items = body.get("items", [])

    from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import KernelProfiler

    results: list[dict[str, Any]] = []
    for item in items:
        try:
            workdir = _make_workdir("pf_")
            kernel_file = _write_tmp(item["kernel_code"], "kernel.py", workdir)
            problem_file = _write_tmp(item["problem_code"], "problem.py", workdir)
            round_num = int(item.get("round_num", 0))
            max_retries = int(item.get("max_retries", 2))
            ncu_bin_path = item.get("ncu_bin_path")

            profiler = KernelProfiler(
                logger=logger,
                artifacts_dir=workdir,
                logs_dir=workdir,
                ncu_bin_path=ncu_bin_path,
                profiling_semaphore=None,
            )
            pr = profiler.profile_kernel(
                kernel_file, problem_file, round_num, max_retries=max_retries
            )
            metrics = pr.metrics if pr is not None else None
            results.append({"metrics": metrics, "error": None})
        except Exception as e:  # noqa: BLE001
            results.append({"metrics": None, "error": str(e)})
    return JSONResponse({"results": results})


# --- main --------------------------------------------------------------


def _resolve_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    if args.token_file:
        return Path(args.token_file).read_text().strip()
    env = os.environ.get("KA_DAEMON_TOKEN")
    if env:
        return env
    # No token anywhere: generate an ephemeral one and print it so the
    # deployer can capture it. (Open mode would be a security hole.)
    tok = secrets.token_urlsafe(24)
    logger.warning("No token configured; generated ephemeral token (print below).")
    print(f"KA_DAEMON_TOKEN={tok}", flush=True)
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelAgent remote GPU daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("KA_DAEMON_PORT", "8765")))
    parser.add_argument("--token", default=None, help="auth token (or use --token-file / KA_DAEMON_TOKEN)")
    parser.add_argument("--token-file", default=None, help="file containing the auth token")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    global _TOKEN
    _TOKEN = _resolve_token(args)

    logger.info("remote daemon starting on %s:%s (auth=%s)", args.host, args.port, bool(_TOKEN))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
