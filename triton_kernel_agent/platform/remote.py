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

"""Remote (HTTP daemon) implementations of the GPU-touching platform
interfaces.

These route test execution, benchmarking, and NCU profiling to a
FastAPI daemon running on a GPU box (typically reached over an ssh
tunnel). The daemon reuses the existing ``_run_test_multiprocess`` /
``Benchmark`` / ``KernelProfiler`` modules, so local and remote
behaviour stay consistent.

Connection info (``url`` / ``token`` / ``timeout``) is injected per-
component via the registry's dict-spec config form::

    platform:
      verifier: {impl: remote, url: "http://127.0.0.1:8765", token: "..."}
      benchmarker: {impl: remote, url: "...", token: "..."}
      profiler: {impl: remote, url: "...", token: "..."}

The non-GPU interfaces (accelerator specs, roofline, bottleneck, RAG)
are intentionally NOT implemented here — they stay on the local agent
brain, which is the whole point of route-2's hybrid design.

Failure semantics mirror the nvidia/noop implementations: when the
daemon is unreachable or errors, calls degrade to safe values
(``False`` / ``float("inf")`` / ``None``) and log, rather than
crashing the worker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from triton_kernel_agent.platform.interfaces import (
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
)

logger = logging.getLogger(__name__)


def _import_verification_worker():
    """Lazy import of ``VerificationWorker`` to avoid pulling the worker
    tree (and its deps) at registry-registration / module-import time.

    ``RemoteVerificationWorker`` subclasses it, but the import only needs
    to resolve when a remote worker is actually constructed.
    """
    from triton_kernel_agent.worker import VerificationWorker

    return VerificationWorker


class _RemoteClient:
    """Thin HTTP client for the remote GPU daemon.

    All endpoints are batch-shaped (``/<op>_batch`` accept a list of
    items and return a list of results). Single-candidate callers send a
    one-element list; the orchestrator can later pack a whole round into
    one request to amortise round-trips (step 6 of the plan).
    """

    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
    ) -> None:
        # Normalise: strip trailing slash so f"{url}/{path}" is clean.
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def post_batch(self, path: str, items: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """POST a batch and return the ``results`` list, or None on failure."""
        endpoint = f"{self.url}/{path.lstrip('/')}"
        payload = {"items": items, "token": self.token}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    endpoint, json=payload, headers=self._headers, timeout=self.timeout
                )
                if resp.status_code == 401:
                    logger.error("remote daemon rejected token (401) at %s", endpoint)
                    return None
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
            except Exception as e:  # noqa: BLE001 - degrade, don't crash
                last_err = e
                # Intermediate retries are noisy; log at debug so a flaky
                # daemon doesn't flood the worker log. The final give-up
                # below is the actionable line.
                logger.debug(
                    "remote POST %s failed (attempt %d/%d): %s",
                    path, attempt + 1, self.max_retries + 1, e,
                )
        logger.error("remote POST %s gave up after %d tries: %s", path, self.max_retries + 1, last_err)
        return None

    def get(self, path: str) -> dict[str, Any] | None:
        endpoint = f"{self.url}/{path.lstrip('/')}"
        try:
            resp = requests.get(endpoint, headers=self._headers, timeout=min(self.timeout, 30.0))
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("remote GET %s failed: %s", path, e)
            return None

    def health(self) -> bool:
        return bool(self.get("health"))


def _read_text(maybe_path: Path | str) -> str:
    """Read a file to text if given an existing path, else return as-is.

    The platform interfaces pass ``problem_file: Path``; the daemon wants
    the problem *content* over the wire (it has its own filesystem). We
    accept either to keep callers simple. In tests / dry-runs the path may
    not exist; fall back to the string form rather than raising.
    """
    if isinstance(maybe_path, Path) and maybe_path.exists():
        return maybe_path.read_text(encoding="utf-8")
    return str(maybe_path)


class RemoteVerifier(KernelVerifier):
    """Verifies kernel correctness by running tests on the remote daemon."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 600.0, **_: Any) -> None:
        self._client = _RemoteClient(url=url, token=token, timeout=timeout)

    def verify(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: list[str],
    ) -> bool:
        problem_code = _read_text(problem_file)
        items = [
            {
                "kernel_code": kernel_code,
                "problem_code": problem_code,
                "test_code": test_code,
            }
        ]
        results = self._client.post_batch("run_test_batch", items)
        if not results:
            logger.error("remote verify: no result from daemon")
            return False
        return bool(results[0].get("success", False))


class RemoteBenchmarker(KernelBenchmarker):
    """Benchmarks kernels / references on the remote daemon (CUDA events)."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 600.0,
        warmup: int = 25,
        repeat: int = 100,
        **_: Any,
    ) -> None:
        self._client = _RemoteClient(url=url, token=token, timeout=timeout)
        self._warmup = warmup
        self._repeat = repeat

    def benchmark_kernel(self, kernel_code: str, problem_file: Path) -> float:
        items = [
            {
                "kernel_code": kernel_code,
                "problem_code": _read_text(problem_file),
                "warmup": self._warmup,
                "repeat": self._repeat,
                "kind": "kernel",
            }
        ]
        results = self._client.post_batch("benchmark_batch", items)
        if not results:
            return float("inf")
        t = results[0].get("time_ms")
        return float(t) if t is not None else float("inf")

    def benchmark_reference(self, problem_file: Path) -> float:
        return self._bench_ref(problem_file, kind="eager")

    def benchmark_reference_compiled(self, problem_file: Path) -> float:
        return self._bench_ref(problem_file, kind="compiled")

    def _bench_ref(self, problem_file: Path, kind: str) -> float:
        items = [
            {
                "problem_code": _read_text(problem_file),
                "warmup": self._warmup,
                "repeat": self._repeat,
                "kind": kind,
            }
        ]
        results = self._client.post_batch("benchmark_batch", items)
        if not results:
            return float("inf")
        t = results[0].get("time_ms")
        return float(t) if t is not None else float("inf")


class RemoteKernelProfiler(KernelProfilerBase):
    """Profiles a kernel via NCU on the remote daemon."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 900.0,
        ncu_bin_path: str | None = None,
        ncu_timeout_seconds: int | None = None,
        **_: Any,
    ) -> None:
        self._client = _RemoteClient(url=url, token=token, timeout=timeout)
        self._ncu_bin_path = ncu_bin_path
        self._ncu_timeout_seconds = ncu_timeout_seconds

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        # kernel_file is a file on the local machine; send its contents.
        kernel_code = _read_text(kernel_file)
        items = [
            {
                "kernel_code": kernel_code,
                "problem_code": _read_text(problem_file),
                "round_num": round_num,
                "max_retries": max_retries,
                "ncu_bin_path": self._ncu_bin_path,
                "ncu_timeout_seconds": self._ncu_timeout_seconds,
            }
        ]
        results = self._client.post_batch("profile_batch", items)
        if not results:
            return None
        metrics = results[0].get("metrics")
        if metrics is None:
            return None
        # Wrap so the result has a `.metrics` attribute, matching the
        # ProfilerResults protocol the orchestrator expects.
        return _RemoteProfilerResult(metrics=metrics, raw=results[0])


class _RemoteProfilerResult:
    """Minimal ProfilerResults-protocol wrapper for remote NCU metrics."""

    def __init__(self, metrics: dict[str, Any], raw: dict[str, Any]) -> None:
        self.metrics = metrics
        self.raw = raw


class RemoteVerificationWorker(_import_verification_worker()):  # type: ignore[misc, valid-type]
    """``VerificationWorker`` that runs tests on the remote daemon.

    This is the route-2 (hybrid) split point: the refinement loop and
    LLM calls stay on the local agent brain (inherited unchanged from
    ``VerificationWorker.verify_with_refinement``), while the *only*
    GPU-touching step — running the test files — is delegated to the
    daemon via ``POST /run_test_batch``.

    We override just ``_single_verification_pass``: it keeps the local
    static ``_detect_pytorch_compute`` check, then ships
    (kernel_code, problem_code, test_code) to the daemon instead of
    spawning a local subprocess. Everything above (the refine loop,
    ``_refine_kernel``→``_call_llm``, history logging) is inherited
    verbatim, so local/remote semantics stay identical apart from where
    the test process physically runs.

    The worker's ``workdir`` already contains ``problem.py`` (copied in
    by the worker runner) and the written test files; we read their
    contents to send over the wire.
    """

    def __init__(self, url: str, token: str | None = None, timeout: float = 600.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._remote_client = _RemoteClient(url=url, token=token, timeout=timeout)

    def _single_verification_pass(self, kernel_code: str) -> tuple[bool, str, str, str | None]:
        """Override: run tests on the remote daemon instead of locally."""
        # Local static check first — no GPU needed, and matches base behaviour.
        violation = self._detect_pytorch_compute(kernel_code)
        if violation:
            message = f"Disallowed PyTorch usage detected: {violation}"
            self.logger.error(message)
            return False, "", message, message

        # Gather file contents from the workdir to ship to the daemon.
        problem_path = self.workdir / "problem.py"
        problem_code = problem_path.read_text(encoding="utf-8") if problem_path.exists() else ""
        test_codes = [
            tf.read_text(encoding="utf-8") for tf in self.test_files if tf.exists()
        ]

        items = [
            {
                "kernel_code": kernel_code,
                "problem_code": problem_code,
                "test_code": test_codes,
            }
        ]
        results = self._remote_client.post_batch("run_test_batch", items)
        if not results:
            err = "remote daemon unreachable; cannot verify"
            self.logger.error(err)
            return False, "", err, None

        success = bool(results[0].get("success", False))
        stdout = results[0].get("stdout", "")
        stderr = results[0].get("stderr", "")
        return success, stdout, stderr, None


