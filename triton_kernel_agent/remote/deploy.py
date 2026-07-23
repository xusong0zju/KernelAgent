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

"""Auto-deployer for the remote GPU daemon (route 2).

One command drives the whole remote side, so the user never copies
anything to the cloud box by hand:

  python -m triton_kernel_agent.remote.deploy \\
      --ssh-host connect.nmb1.seetacloud.com --ssh-port 10839 \\
      --ssh-user root --ssh-pass '...' \\
      --repo /home/xusong/PreResearch_WS/KernelAgent

It will, on the cloud box:
  1. sync the local repo (tar over ssh — no rsync dependency)
  2. ensure a Python venv + ``pip install -e .`` (+ deps) if missing
  3. write a daemon token
  4. launch the daemon (nohup, background) on an ephemeral port
  5. poll /health until ready

Then locally:
  6. open an ssh tunnel (127.0.0.1:<local> -> 127.0.0.1:<remote_port>)
  7. print the ``platform:`` YAML snippet the user pastes into their
     config (or the env vars the agent reads).

``teardown()`` reverses: kill daemon + close tunnel.

Security: credentials come from CLI args or KA_REMOTE_SSH_* env, never
hardcoded. The daemon token is generated locally and shipped to the
remote ``.env.daemon``; the daemon binds 127.0.0.1 only, so the tunnel
is the sole ingress.

This is intentionally dependency-light: only stdlib + sshpass (already
used elsewhere in this user's setup). No paramiko/fabric.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("remote.deploy")

_DEFAULT_REMOTE_DIR = "kernelagent_remote"
_DEFAULT_PORT = 8765


@dataclass
class RemoteHandle:
    """Handle to a running remote daemon + its local tunnel."""

    ssh_target: str  # user@host (port via -p elsewhere)
    ssh_port: int
    ssh_user: str
    ssh_host: str
    remote_port: int
    local_port: int
    token: str
    tunnel_proc: subprocess.Popen | None = None

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def platform_snippet(self) -> str:
        """YAML platform: block to paste into a KernelAgent config."""
        spec = f'{{impl: remote, url: "{self.local_url}", token: "{self.token}"}}'
        return (
            "platform:\n"
            f"  verifier: {spec}\n"
            f"  benchmarker: {spec}\n"
            f"  profiler: {spec}\n"
            f"  verification_worker: {spec}\n"
        )


# ---------------------------------------------------------------------------
# ssh plumbing
# ---------------------------------------------------------------------------


def _ssh_base_cmd(host: str, port: int, user: str, password: str | None) -> list[str]:
    """Build an ssh command prefix that authenticates non-interactively.

    Uses sshpass when a password is given (this user's cloud box is
    password-auth); falls back to plain ssh (key auth) otherwise.
    """
    if password:
        return [
            "sshpass", "-p", password,
            "ssh",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-p", str(port),
            f"{user}@{host}",
        ]
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",
        "-p", str(port),
        f"{user}@{host}",
    ]


def _run_remote(host, port, user, password, script: str, timeout: int = 300) -> tuple[int, str]:
    """Run a shell script on the remote box; return (exitcode, combined output)."""
    cmd = _ssh_base_cmd(host, port, user, password) + ["bash", "-lc", script]
    logger.debug("remote cmd: %s", " ".join(shlex.quote(c) for c in cmd[:1]) + " …")
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout + r.stderr)


def _sync_repo_tar(
    host, port, user, password, local_repo: Path, remote_dir: str,
) -> None:
    """Sync local_repo -> remote_dir via tar over ssh (no rsync needed).

    Implemented as a single shell pipeline (``tar ... | ssh ... 'tar x'``)
    so the OS handles the pipe; managing two Popen objects' stdin/stdout
    in Python is brittle (fd lifetime / flush ordering).
    """
    local_repo = local_repo.resolve()
    excludes = [".git", "triton_kernel_logs", "__pycache__", "*.pyc",
                ".env", "node_modules", ".venv", "env", "build", "install"]
    exclude_args = ""
    for e in excludes:
        exclude_args += f" --exclude {shlex.quote(e)}"

    ssh_prefix = " ".join(_ssh_base_cmd(host, port, user, password))
    # NOTE: the inner tar runs on the REMOTE (after ssh), the outer tar locally.
    pipeline = (
        f"tar c -C {shlex.quote(str(local_repo))}{exclude_args} . | "
        f"{ssh_prefix} "
        f"'mkdir -p {shlex.quote(remote_dir)} && tar x -C {shlex.quote(remote_dir)} && echo SYNC_DONE'"
    )
    logger.info("syncing repo %s -> %s:%s (tar over ssh)", local_repo, host, remote_dir)
    r = subprocess.run(["bash", "-c", pipeline], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"repo sync failed (exit {r.returncode}):\n{r.stderr}\n{r.stdout}")
    if "SYNC_DONE" not in r.stdout:
        raise RuntimeError(f"repo sync produced no SYNC_DONE:\n{r.stdout}\n{r.stderr}")
    logger.info("repo synced")


# ---------------------------------------------------------------------------
# Deploy steps
# ---------------------------------------------------------------------------


_SETUP_SCRIPT = """set -e
cd {remote_dir}
# Pick a Python that can already import torch (cloud GPU images usually
# have torch in a conda base env). Prefer ~/miniconda3/bin/python, then
# python3. Fall back to python3 even without torch (setup will try to
# install deps). Export as PYBIN for the rest of the script.
for cand in "$HOME/miniconda3/bin/python" "$(command -v python3)"; do
  if [ -n "$cand" ] && "$cand" -c 'import torch' >/dev/null 2>&1; then
    PYBIN="$cand"; break
  fi
done
: ${{PYBIN:=$(command -v python3)}}
echo "using PYBIN=$PYBIN"
# Ensure pip available.
"$PYBIN" -m pip --version >/dev/null 2>&1 || "$PYBIN" -m ensurepip --upgrade
# Install the package + the 'remote' extra (fastapi/uvicorn). If the
# full install fails (e.g. some heavy dep won't build), retry --no-deps
# so the daemon at least imports — torch/triton are already present.
"$PYBIN" -m pip install -e '.[remote]' || "$PYBIN" -m pip install -e . --no-deps
"$PYBIN" -m pip install fastapi uvicorn >/dev/null 2>&1 || true
"$PYBIN" -c "import torch, triton, fastapi, uvicorn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| triton', triton.__version__, '| fastapi', fastapi.__version__)"
echo "PYBIN=$PYBIN"
"""


def _ensure_env_and_deps(host, port, user, password, remote_dir: str, python_bin: str | None = None) -> str:
    """Ensure remote repo installed editable; return the remote python bin path.

    If ``python_bin`` is given (an absolute remote path), use it directly
    instead of probing — useful when the auto-detected env has a known
    incompatibility (e.g. triton 3.3 on py3.12 hits ``PY_SSIZE_T_CLEAN``).
    """
    if python_bin:
        # Use the given python directly; just ensure pip + install the package.
        script = (
            f"set -e\ncd {shlex.quote(remote_dir)}\n"
            f'PYBIN={shlex.quote(python_bin)}\n'
            f'"$PYBIN" -m pip --version >/dev/null 2>&1 || "$PYBIN" -m ensurepip --upgrade\n'
            f'"$PYBIN" -m pip install -e ".[remote]" || "$PYBIN" -m pip install -e . --no-deps\n'
            f'"$PYBIN" -c "import torch, triton, fastapi, uvicorn; '
            f'print(\'torch\', torch.__version__, \'cuda\', torch.cuda.is_available(), '
            f"'| triton', triton.__version__, '| fastapi', fastapi.__version__)\"\n"
            f'echo "PYBIN=$PYBIN"\n'
        )
    else:
        script = _SETUP_SCRIPT.format(remote_dir=shlex.quote(remote_dir))
    code, out = _run_remote(host, port, user, password, script, timeout=900)
    if code != 0:
        raise RuntimeError(f"remote setup failed (exit {code}):\n{out}")
    logger.info("remote env ready:\n%s", out.strip())
    # Extract the PYBIN the script chose (last "PYBIN=..." line).
    pybin = "python3"
    for line in out.splitlines():
        if line.startswith("PYBIN="):
            pybin = line.split("=", 1)[1].strip()
    return pybin or "python3"


def _launch_daemon(
    host, port, user, password, remote_dir: str, pybin: str,
    daemon_port: int, token: str,
) -> None:
    """Launch the daemon on the remote in the background; write token file.

    Uses ``bash -lc`` with a single here-doc'd script so $HOME/cd expand
    consistently (mixing ``cat > "$HOME/..."`` with inline args was
    brittle: ``_run_remote`` passes the whole script as one ``bash -lc``
    arg, and quote-interaction ate the ``$HOME`` in some paths).
    """
    # remote_dir is relative to $HOME (e.g. "kernelagent_remote").
    rd = remote_dir.lstrip("/")
    token_file = f"$HOME/{rd}/.env.daemon"
    daemon_script = f"$HOME/{rd}/scripts/remote_daemon.py"
    log_file = f"$HOME/{rd}/daemon.log"
    # Write the whole launch as ONE bash script (single-quoted here-doc)
    # so no shell quoting of the outer command interferes.
    script = (
        "set -e\n"
        f'cat > "{token_file}" <<\'KA_TOKEN_EOF\'\n'
        f"{token}\n"
        "KA_TOKEN_EOF\n"
        f'cd "$HOME/{rd}"\n'
        f'nohup {pybin} "{daemon_script}" '
        f'--port {daemon_port} --token-file "{token_file}" '
        f'--log-level INFO > "{log_file}" 2>&1 &\n'
        'disown\n'
        'echo LAUNCHED'
    )
    code, out = _run_remote(host, port, user, password, script, timeout=60)
    if code != 0:
        raise RuntimeError(f"daemon launch failed:\n{out}")
    logger.info("remote daemon launched: %s", out.strip())


def _wait_health(url: str, token: str, tries: int = 40, delay: float = 1.0) -> dict:
    import requests

    last = None
    for _ in range(tries):
        try:
            r = requests.get(url + "/health",
                             headers={"Authorization": f"Bearer {token}"}, timeout=5)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:120]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(delay)
    raise RuntimeError(f"daemon /health never came up at {url}: {last}")


def _open_tunnel(host, port, user, password, local_port, remote_port) -> subprocess.Popen:
    """Open ssh -L tunnel in the background; return the Popen."""
    cmd: list[str]
    if password:
        cmd = ["sshpass", "-p", password, "ssh",
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
               "-N",  # no remote command; just forward
               "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
               "-p", str(port), f"{user}@{host}"]
    else:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
               "-N", "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
               "-p", str(port), f"{user}@{host}"]
    logger.info("opening tunnel 127.0.0.1:%d -> %s:%d", local_port, host, remote_port)
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    # Give it a moment to establish.
    time.sleep(1.5)
    if p.poll() is not None:
        err = p.stderr.read()
        raise RuntimeError(f"tunnel failed to start: {err}")
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deploy(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_pass: str | None,
    repo: Path,
    remote_dir: str = _DEFAULT_REMOTE_DIR,
    daemon_port: int | None = None,
    local_port: int | None = None,
    python_bin: str | None = None,
) -> RemoteHandle:
    """Deploy + launch daemon + open tunnel. Returns a RemoteHandle.

    ``python_bin`` (absolute remote path) overrides the auto-detected
    python; use it when the default env is incompatible (e.g. triton on
    py3.12).
    """
    token = secrets.token_urlsafe(24)
    daemon_port = daemon_port or _DEFAULT_PORT
    local_port = local_port or daemon_port

    # Kill any prior daemon on that port so a fresh deploy binds cleanly
    # (otherwise the new process dies on "address already in use" and the
    # tunnel reaches the STALE daemon — token mismatch → 401).
    try:
        _run_remote(
            ssh_host, ssh_port, ssh_user, ssh_pass,
            f"pkill -f 'remote_daemon.py --port {daemon_port}' || true; "
            f"sleep 1",
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("pre-deploy pkill failed (continuing): %s", e)

    # 1. sync code
    _sync_repo_tar(ssh_host, ssh_port, ssh_user, ssh_pass, repo, remote_dir)
    # 2. ensure deps
    pybin = _ensure_env_and_deps(ssh_host, ssh_port, ssh_user, ssh_pass, remote_dir, python_bin)
    # 3+4. launch daemon
    _launch_daemon(ssh_host, ssh_port, ssh_user, ssh_pass, remote_dir, pybin,
                   daemon_port, token)
    # 5. open tunnel
    tunnel = _open_tunnel(ssh_host, ssh_port, ssh_user, ssh_pass, local_port, daemon_port)
    handle = RemoteHandle(
        ssh_target=f"{ssh_user}@{ssh_host}", ssh_port=ssh_port, ssh_user=ssh_user,
        ssh_host=ssh_host, remote_port=daemon_port, local_port=local_port, token=token,
        tunnel_proc=tunnel,
    )
    # 6. wait for health THROUGH the tunnel (proves both daemon + tunnel work)
    health = _wait_health(handle.local_url, token)
    logger.info("daemon healthy: %s", health)
    return handle


def teardown(handle: RemoteHandle) -> None:
    """Kill the remote daemon and close the local tunnel."""
    if handle.tunnel_proc and handle.tunnel_proc.poll() is None:
        handle.tunnel_proc.terminate()
        try:
            handle.tunnel_proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            handle.tunnel_proc.kill()
    # Kill the remote daemon by port.
    _run_remote(
        handle.ssh_host, handle.ssh_port, handle.ssh_user, None,
        f"pkill -f 'remote_daemon.py --port {handle.remote_port}' || true",
        timeout=30,
    )
    logger.info("torn down (daemon on port %d + tunnel)", handle.remote_port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _from_env_or_args(args) -> tuple[str, int, str, str | None]:
    host = args.ssh_host or os.environ.get("KA_REMOTE_SSH_HOST")
    port = args.ssh_port or int(os.environ.get("KA_REMOTE_SSH_PORT", "22"))
    user = args.ssh_user or os.environ.get("KA_REMOTE_SSH_USER", "root")
    pwd = args.ssh_pass or os.environ.get("KA_REMOTE_SSH_PASS")
    if not host:
        raise SystemExit("ssh host required (--ssh-host or KA_REMOTE_SSH_HOST)")
    return host, port, user, pwd


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy the remote GPU daemon")
    ap.add_argument("--ssh-host", default=None)
    ap.add_argument("--ssh-port", type=int, default=None)
    ap.add_argument("--ssh-user", default=None)
    ap.add_argument("--ssh-pass", default=None)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--remote-dir", default=_DEFAULT_REMOTE_DIR)
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT, help="daemon (and default local) port")
    ap.add_argument("--local-port", type=int, default=None)
    ap.add_argument("--teardown", action="store_true", help="kill an existing daemon + tunnel")
    ap.add_argument("--python-bin", default=None,
                    help="absolute remote python path (e.g. ~/miniconda3/envs/ka_gpu/bin/python); "
                         "overrides auto-detect — use when default env is incompatible")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host, port, user, pwd = _from_env_or_args(args)

    if args.teardown:
        # Best-effort teardown by port: no handle object kept across runs.
        h = RemoteHandle(ssh_target=f"{user}@{host}", ssh_port=port, ssh_user=user,
                         ssh_host=host, remote_port=args.port, local_port=args.local_port or args.port,
                         token="")
        teardown(h)
        print("teardown done")
        return

    handle = deploy(host, port, user, pwd, Path(args.repo),
                    remote_dir=args.remote_dir, daemon_port=args.port,
                    local_port=args.local_port, python_bin=args.python_bin)
    print("\n========================================")
    print("Remote GPU daemon is UP. Paste this into your config:")
    print("========================================")
    print(handle.platform_snippet())
    print("Local URL:", handle.local_url)
    print("Token:", handle.token)


if __name__ == "__main__":
    main()
