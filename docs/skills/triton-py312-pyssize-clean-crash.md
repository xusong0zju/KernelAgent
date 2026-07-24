---
name: triton-py312-pyssize-clean-crash
description: |
  Fix the `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`
  crash that happens the first time a Triton `@triton.jit` kernel is compiled
  and launched under Python 3.12 with older Triton (notably triton 3.3.x).
  Use when: (1) a Triton kernel launch (`kernel[grid](...)`) raises this
  SystemError with the stack going through
  `triton/compiler/compiler.py _init_handles → driver.active.utils.load_binary`,
  (2) `torch.cuda.is_available()` is True and CUDA context works but Triton
  JIT compile crashes, (3) you installed torch+triton into a conda base env
  that happens to be Python 3.12. Covers: the CPython C-API PY_SSIZE_T_CLEAN
  requirement, why older triton's compiled extension trips it on 3.12, and
  the reliable fix (use a Python 3.11 env with a triton version that matches
  its build target).
author: Claude Code
version: 1.0.0
date: 2026-07-23
---

# Triton crash on Python 3.12: PY_SSIZE_T_CLEAN

## Problem

You have a working CUDA setup (`torch.cuda.is_available()` is True, you
can allocate tensors on GPU), but the **first** `@triton.jit` kernel you
compile+launch crashes with:

```
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
```

Stack (truncated):
```
File ".../your_kernel.py", in kernel_function
    add_one_kernel[grid](x, out, n, BLOCK_SIZE=16)
File ".../triton/runtime/jit.py", in <lambda>
File ".../triton/runtime/jit.py", in run
File ".../triton/compiler/compiler.py", in __getattribute__
    self._init_handles()
File ".../triton/compiler/compiler.py", in _init_handles
    self.module, self.function, ... = driver.active.utils.load_binary(...)
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
```

The crash is NOT in your kernel code — it is inside Triton's compiled C
extension (`load_binary`) the moment it tries to load the freshly-compiled
kernel module. Eager CUDA / torch CUDA ops work fine; only Triton JIT breaks.

## Context / Trigger Conditions

- Python **3.12.x** (check `python --version`)
- Triton **3.x** built against an older Python target — confirmed reproducer:
  `torch 2.7.0+cu128` + `triton 3.3.0` on `Python 3.12.3` (a stock conda
  base env on a cloud GPU image)
- First Triton kernel compile in the process. Restarting the process and
  hitting it again reproduces 100% (it is not a stale-cache artefact).

## Root Cause

CPython's C API requires the `PY_SSIZE_T_CLEAN` macro to be defined before
including `Python.h` for any function taking `#`-format args (it changes
`int` sizes to `Py_ssize_t`). From Python 3.12 this is enforced more
strictly; a compiled C extension that was built against an older Python
target (or without the macro) trips `SystemError` when it calls such an
API at runtime.

Triton's `load_binary` (the compiled loader extension) in some 3.x
release-line builds does not satisfy this on 3.12, so loading the JIT
kernel module raises the error. The bug is in the **mismatch between the
Triton build target and the running Python 3.12** — not your code, not
CUDA, not torch.

## Solution

**Use Python 3.11 (or 3.10) with a triton version known-good for that
Python.** This is the only reliable fix; there is no runtime workaround.

### Confirm it is this bug before fixing

```bash
python --version          # expect 3.12.x
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
# True for cuda, but:
python -c "
import torch, triton, triton.language as tl
@triton.jit
def k(p, n, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0,B); m = i<n
    tl.store(p+i, tl.load(p+i, mask=m)+1, mask=m)
x = torch.zeros(64, device='cuda')
k[(1,)](x, 64, B=16)
print('ok')
"   # expect the SystemError
```

### Fix: create a 3.11 env and use it for anything that runs Triton

```bash
# one-time
conda create -n ka_gpu python=3.11 -y
conda run -n ka_gpu pip install torch triton numpy

# verify (no SystemError)
conda run -n ka_gpu python -c "import torch,triton,triton.language as tl;\
@triton.jit
def k(p,n,B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<n
    tl.store(p+i,tl.load(p+i,mask=m)+1,mask=m)
x=__import__('torch').zeros(64,device='cuda'); k[(1,)](x,64,B=16); print('ok')"

# then run everything that touches Triton with this env's python, e.g.:
~/miniconda3/envs/ka_gpu/bin/python your_script.py
```

Confirmed-good combo on an RTX 2080 Ti (Turing, CUDA 12.8 driver 595):
`Python 3.11` + `torch 2.13.0+cu130` + `triton 3.7.1` — Triton kernels
compile and run correctly. (Use `pip install torch triton` and let pip
resolve; pinning to a known-good combo avoids the issue.)

### If you must stay on 3.12

Upgrade triton to the newest version that explicitly supports 3.12
(`pip install -U triton`) and hope the build target moved — this is
hit-or-miss and was NOT reliable for 3.3.0. The 3.11 env is the sure fix.

## Verification

After switching to the 3.11 env, the reproducer above prints `ok` instead
of raising. A real `@triton.jit` kernel launch returns correct output and
no `SystemError`. `torch.cuda` works as before (it was never the problem).

## Example

A remote GPU daemon was launched on a cloud box using the box's default
`~/miniconda3/bin/python` (3.12 + torch2.7+cu128 + triton3.3). `/health`
worked, `nvidia-smi` worked, even a plain `torch` kernel worked — but any
`@triton.jit` kernel via the daemon's `/run_test_batch` crashed with this
`SystemError`. Fix: deploy the daemon with `--python-bin
/root/miniconda3/envs/ka_gpu/bin/python` (a 3.11 env). Same kernels then
compiled and ran correctly. (See KernelAgent `docs/远程GPU部署.md`坑1.)

## Notes

- This is a **build-target / ABI** issue, not a logic bug — the kernel
  source is fine. Don't waste time rewriting the kernel.
- It only manifests on the **first** Triton compile per process; subsequent
  compiles in the same process may appear to work if the bad handle is
  cached, but a fresh process reproduces.
- `torch` CUDA ops and `torch.compile`-inductor paths may or may not use
  the same Triton code path — don't assume; test a raw `@triton.jit` launch.
- If you see this inside a long-running daemon, remember the daemon process
  must be started with the 3.11 python, not just your script.

## References

- CPython `PY_SSIZE_T_CLEAN`: enforced since the API was tightened; from
  3.12 the `#` format without the macro raises `SystemError`.
- Confirmed empirically on Triton 3.3.0 / Python 3.12.3 (cloud GPU image),
  fixed by Python 3.11 + triton 3.7.1. Verify against your installed
  versions (`python -c "import sys,triton,torch;print(sys.version,
  triton.__version__, torch.__version__)"`).
