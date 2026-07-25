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

# Triton 在 Python 3.12 上崩溃：PY_SSIZE_T_CLEAN

## 问题

CUDA 环境正常（`torch.cuda.is_available()` 为 True、能在 GPU 上分配 tensor），但你**第一次**编译+启动 `@triton.jit` kernel 就崩：

```
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
```

栈（截断）：
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

崩的不在你的 kernel 代码——是 Triton 编译好的 C 扩展（`load_binary`）加载刚编译的 kernel 模块时崩。eager CUDA / torch CUDA 算子正常，只有 Triton JIT 崩。

## 触发条件

- Python **3.12.x**（`python --version` 确认）
- Triton **3.x** 按较老 Python target 构建——确认复现：`torch 2.7.0+cu128` + `triton 3.3.0` on `Python 3.12.3`（云 GPU 镜像默认 conda base env）
- 进程内**第一次** Triton kernel 编译。重启进程再触发 100% 复现（不是缓存残留）。

## 根因

CPython C API 要求 include `Python.h` 前定义 `PY_SSIZE_T_CLEAN` 宏（把 `#`-格式参数的 `int` 改成 `Py_ssize_t`）。Python 3.12 起强制更严；按较老 Python target 构建（或没定义该宏）的编译 C 扩展，运行时调这类 API 就触发 `SystemError`。

Triton 的 `load_binary`（编译好的 loader 扩展）在部分 3.x 构建里不满足 3.12 要求，所以加载 JIT kernel 模块时报错。bug 在 **Triton 构建目标 vs 运行 Python 3.12 的不匹配**——不是你的代码、不是 CUDA、不是 torch。

## 解决

**用 Python 3.11（或 3.10）+ 该 Python 下已知可用的 triton 版本。** 这是唯一可靠修法，无运行时 workaround。

### 修前确认是这个 bug

```bash
python --version          # 期望 3.12.x
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
# cuda 为 True，但：
python -c "
import torch, triton, triton.language as tl
@triton.jit
def k(p, n, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0,B); m = i<n
    tl.store(p+i, tl.load(p+i, mask=m)+1, mask=m)
x = torch.zeros(64, device='cuda')
k[(1,)](x, 64, B=16)
print('ok')
"   # 期望那个 SystemError
```

### 修法：建 3.11 环境跑所有碰 Triton 的东西

```bash
# 一次性
conda create -n ka_gpu python=3.11 -y
conda run -n ka_gpu pip install torch triton numpy

# 验证（无 SystemError）
conda run -n ka_gpu python -c "import torch,triton,triton.language as tl;\
@triton.jit
def k(p,n,B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<n
    tl.store(p+i,tl.load(p+i,mask=m)+1,mask=m)
x=__import__('torch').zeros(64,device='cuda'); k[(1,)](x,64,B=16); print('ok')"

# 之后所有碰 Triton 的都用这个 env 的 python，如：
~/miniconda3/envs/ka_gpu/bin/python your_script.py
```

RTX 2080 Ti（Turing, CUDA 12.8 驱动 595）上确认可用组合：`Python 3.11` + `torch 2.13.0+cu130` + `triton 3.7.1`——Triton kernel 正常编译运行。（`pip install torch triton` 让 pip 解析；锁已知可用组合可避坑。）

### 若必须留 3.12

升级 triton 到明确支持 3.12 的最新版（`pip install -U triton`）碰运气——不靠谱，3.3.0 不行。3.11 env 才是确定修法。

## 验证

切到 3.11 env 后，上面 reproducer 打印 `ok` 而非报错。真实 `@triton.jit` kernel 启动返回正确输出、无 `SystemError`。`torch.cuda` 如常（它本不是问题）。

## 例子

某云机上用默认 `~/miniconda3/bin/python`（3.12 + torch2.7+cu128 + triton3.3）起了个远程 GPU daemon。`/health` 正常、`nvidia-smi` 正常、连普通 `torch` kernel 都正常——但任何经 daemon `/run_test_batch` 的 `@triton.jit` kernel 都崩这个 `SystemError`。修法：用 `--python-bin /root/miniconda3/envs/ka_gpu/bin/python`（3.11 env）部署 daemon。同些 kernel 随后正常编译运行。（见 KernelAgent `docs/远程GPU部署.md` 坑1。）

## 注意

- 这是**构建目标/ABI** 问题，不是逻辑 bug——kernel 源码没问题。别浪费时间重写 kernel。
- 只在进程**第一次** Triton 编译时出现；同进程后续编译若坏 handle 被缓存可能看似正常，但新进程必复现。
- `torch` CUDA 算子和 `torch.compile`-inductor 路径可能用、也可能不用同一 Triton 代码路径——别假设；直接测个原生 `@triton.jit` 启动。
- 长驻 daemon 里看到这个，记得**daemon 进程**要用 3.11 python 启动，不只是你的脚本。

## 参考

- CPython `PY_SSIZE_T_CLEAN`：API 收紧后强制；3.12 起 `#` 格式无宏会抛 `SystemError`。
- 实测：Triton 3.3.0 / Python 3.12.3（云 GPU 镜像），Python 3.11 + triton 3.7.1 修复。以你装的版本为准验证（`python -c "import sys,triton,torch;print(sys.version, triton.__version__, torch.__version__)"`）。
