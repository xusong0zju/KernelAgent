# 远程 GPU daemon 部署与使用（路线 2 · 方案 C）

> 本文记录如何让本机 agent 大脑调度云端 GPU 真跑 Triton kernel 的编译/测试/benchmark/NCU profiling。
> 设计背景见 `docs/为何KernelAgent强于直接用LLM.md`（闭环论证）与本机安装见 `docs/安装与环境交接.md`。
> 写于 2026-07-23，已端到端跑通（bug kernel → 云端验证失败 → 本机 LLM refine → 云端验证通过）。

---

## 一、这是什么 / 为什么需要

本机无 GPU（驱动太老 + 无设备），KernelAgent 之前只能验证到"LLM 生成阶段"——kernel 代码能生成，但编不编得过、跑得对不对、快不快，全都验证不了（`generate_kernel` 的 `success=False` 仅因本机无 GPU）。

**路线 2 = 远程 daemon 改造**：本机保留 agent 大脑（LLM 调用、beam search 编排、瓶颈分析、reflexion），把 3 个 GPU 操作（测试执行 / benchmark 计时 / NCU profiling）下沉到一个常驻云端的 FastAPI daemon，本机通过 ssh 隧道 HTTP 调用。**全程自动，用户不手动拷贝任何东西到云端。**

### 架构

```
本机（agent 大脑）                          云端 GPU 机（执行 daemon）
┌──────────────────────────┐              ┌──────────────────────────┐
│ OptManager (beam search)  │              │ FastAPI daemon (常驻)     │
│  └ spawn N worker 进程     │──ssh 隧道──▶│  POST /run_test_batch     │
│     OptimizationWorker     │  HTTP/JSON  │  POST /benchmark_batch    │
│      ├ LLM refine (本机)   │              │  POST /profile_batch      │
│      ├ bottleneck (本机)  │              │  GET  /health /specs      │
│      └ GPU 操作 → Remote*  │              │ 复用: _run_test_multiproc │
│ RemoteVerificationWorker   │              │  / Benchmark / KernelProf │
│ RemoteBenchmarker          │              │ torch + triton + ncu      │
│ RemoteKernelProfiler       │              └──────────────────────────┘
└──────────────────────────┘
```

**关键设计**：`verify_with_refinement` 是 GPU+LLM 交织循环（跑 test → 失败则 LLM refine → 再跑 test）。方案 C 把它拆开——`RemoteVerificationWorker` 只 override `_single_verification_pass`（test 执行段走 daemon），refine 循环 + LLM 调用**原样继承父类**留在本机。这样 LLM 凭证和完整 agent 逻辑留在本机，只有物理跑 test 去了云端。

---

## 二、5 分钟上手

### 前置

- 本机已装 KernelAgent（`kernel_agent` conda 环境，见安装文档）
- 一台云端 GPU 机，ssh 可达，且云端有能 import torch/triton 的 python（见第四节坑）
- 本机有 `sshpass`（`apt install sshpass`）

### 一条命令部署

```bash
env -u ALL_PROXY -u all_proxy \
~/miniconda3/envs/kernel_agent/bin/python -m triton_kernel_agent.remote.deploy \
  --ssh-host connect.nmb1.seetacloud.com --ssh-port 10839 \
  --ssh-user root --ssh-pass '你的密码' \
  --repo /home/xusong/PreResearch_WS/KernelAgent \
  --port 8765 \
  --python-bin /root/miniconda3/envs/ka_gpu/bin/python
```

它会自动（用户零手动拷贝）：
1. tar 打包本机仓库 → ssh 传到云端 `~/kernelagent_remote/`（排除 .git/logs/cache）
2. 在云端 `pip install -e ".[remote]"`（装 KernelAgent + fastapi/uvicorn）
3. 生成随机 daemon token 写入云端 `.env.daemon`
4. `nohup` 后台拉起 `scripts/remote_daemon.py`（监听 127.0.0.1）
5. 本机建 `ssh -L 127.0.0.1:8765 -> 云端:8765` 隧道
6. 轮询 `/health` 确认就绪

成功后打印一段 `platform:` YAML，**直接粘进你的 KernelAgent config**：

```yaml
platform:
  verifier: {impl: remote, url: "http://127.0.0.1:8765", token: "..."}
  benchmarker: {impl: remote, url: "http://127.0.0.1:8765", token: "..."}
  profiler: {impl: remote, url: "http://127.0.0.1:8765", token: "..."}
  verification_worker: {impl: remote, url: "http://127.0.0.1:8765", token: "..."}
```

### 跑

配好 config 后，原来的命令原样跑，执行层自动走云端：

```bash
env -u ALL_PROXY -u all_proxy OPENAI_MODEL=deepseek-v4-pro \
python -m Fuser.auto_agent --problem examples/optimize_02_rmsnorm/problem.py --config <你的config.yaml>
```

或直接用 Python API（`TritonKernelAgent` / `OptimizationWorker` 配 `platform_config` 指向 remote）。

### 关隧道 / 停 daemon

```bash
~/miniconda3/envs/kernel_agent/bin/python -m triton_kernel_agent.remote.deploy \
  --ssh-host ... --ssh-port ... --ssh-user root --ssh-pass '...' --teardown --port 8765
```

---

## 三、组件清单（改了什么）

| 文件 | 作用 |
|---|---|
| `triton_kernel_agent/platform/remote.py` | `RemoteVerifier` / `RemoteBenchmarker` / `RemoteKernelProfiler` / `RemoteVerificationWorker`。HTTP 调 daemon，不可达时降级到 `False/inf/None` 不崩。 |
| `triton_kernel_agent/platform/registry.py` | `_register_builtins` 追加注册 `"remote"` 系列；`create_from_config` 扩展支持 dict-spec（`{impl: remote, url, token}`）带参，纯字符串形式向后兼容。 |
| `triton_kernel_agent/opt_worker.py` | worker 的 benchmarker / verification_worker 改走 platform 注入（仿 profiler 的 if/else），让 remote 能注入进来。 |
| `scripts/remote_daemon.py` | 云端 FastAPI daemon：`/health` `/specs` `/run_test_batch` `/benchmark_batch` `/profile_batch`，复用现有模块，Bearer token 鉴权，监听 127.0.0.1。 |
| `triton_kernel_agent/remote/deploy.py` | 自动部署器：sync + 装依赖 + 起 daemon + 建隧道 + 健康检查。`--python-bin` 可指定云端 python。 |
| `pyproject.toml` | 加 `[project.optional-dependencies] remote = [fastapi, uvicorn]`。 |
| `tests/test_platform_registry.py` | registry dict-spec 的单元测试（6 个）。 |

**agent 大脑零改动**：OptManager / beam search / 策略 / prompt 模板 / reflexion / bottleneck / roofline 全部不动。

---

## 四、踩坑记录（务必看，非显而易见）

### 坑 1：云端 base conda 的 triton 3.3 + py3.12 跑 Triton kernel 崩

**现象**：daemon 用 `~/miniconda3/bin/python`（py3.12 + torch2.7+cu128 + triton3.3）起，能起服务、`/health` 正常，但一跑真 Triton kernel 就报：
```
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
```
（栈在 `triton/compiler/compiler.py _init_handles → driver.active.utils.load_binary`）

**根因**：triton 3.3 的 Python C 扩展和 py3.12 在 `#` 格式上有已知不兼容。云端 base 环境虽然 torch.cuda 可用，但 triton 编译路径崩。

**解法**：在云端另建一个 py3.11 环境（triton 3.7 + torch 2.13），用 `--python-bin` 指定它起 daemon：
```bash
# 云端建环境（一次性）
~/miniconda3/bin/conda create -n ka_gpu python=3.11 -y
~/miniconda3/envs/ka_gpu/bin/pip install torch triton numpy
# 部署时指定
--python-bin /root/miniconda3/envs/ka_gpu/bin/python
```
deploy.py 会用这个 python 装 KernelAgent 并起 daemon。**RTX 2080 Ti + py3.11 + triton3.7 跑 Triton kernel 正常**。

### 坑 2：ssh 启动后台 daemon 的 shell 引用地狱

deploy.py 早期版本在 ssh 里拼 `bash -lc "cd X && printf %s TOKEN > f && nohup ... &"`，出过三种坑：
- `printf %s <token>` 把 token 当**格式串**，token 里有 `%` 就被吞，token 文件变空
- `bash -lc` 的 cwd 不可靠，daemon 脚本路径解析成 `~/scripts/...` 而非 `~/kernelagent_remote/scripts/...`
- 多层引号 + `$HOME` 在 `bash -lc` 单参里展开不一致

**解法**：把整个启动写成**单个 here-doc 脚本**，路径全用 `$HOME/...` 绝对形式，token 用 `cat > f <<'EOF'` 写入（quoted heredoc 不解释内容）。见 `deploy.py` 的 `_launch_daemon`。

### 坑 3：tar 管道用两个 Popen 管理易爆

sync 代码最初用 `Popen(ssh, stdin=PIPE) + run(tar, stdout=p_ssh.stdin)`，出现 `ValueError: I/O operation on closed file`（tar 进程结束关了 p_ssh.stdin，`communicate` 再 flush 就崩）。

**解法**：整个 sync 写成**一条 shell 管道** `tar ... | ssh ... 'tar x'`，交给 OS 管管道，Python 只 `subprocess.run` 一条。见 `_sync_repo_tar`。

### 坑 4：daemon 返回 inf 不能 JSON 序列化

benchmark 本机无 GPU 时返回 `float("inf")`，FastAPI 序列化报 `Out of range float values are not JSON compliant` → 500。

**解法**：daemon 端 `_json_safe_float` 把 inf/nan 转 `None`；Remote* 客户端把 `None` 还原回 `inf` 语义。

### 坑 5：Benchmark 需要 lock 对象

`Benchmark.__init__` 的 `benchmark_lock` 会 `.acquire()`，传 `None` 崩。daemon 单进程，传一个 `threading.Lock()` 即可（批量串行，锁是冗余但必要）。

---

## 五、端到端验证记录（2026-07-23）

跑了一个最小闭环证明整条链路通：

1. 故意给一个 **bug kernel**（`def kernel_function(x): return x`，忘了 +1）
2. `RemoteVerificationWorker.verify_with_refinement` → 云端 daemon `/run_test_batch` 跑 test → **失败**（`y != x+1`），stderr 回传
3. 本机 LLM（deepseek-v4-pro 经金山云）refine → 生成正确的 `@triton.jit` Triton kernel
4. 云端 daemon 再跑 test → **`success: True`**，`Y: [2.0, 3.0, 4.0] OK: True`

**结论**：本机 agent 大脑（LLM refine）+ 云端 GPU 执行（test 编译运行）的闭环完全打通，kernel 在云端 RTX 2080 Ti 上真实编译运行通过。这正是本机无 GPU 时卡住的环节。

---

## 六、配置速查

### 环境变量（deploy 用）

| 变量 | 作用 |
|---|---|
| `KA_REMOTE_SSH_HOST/PORT/USER/PASS` | deploy 的 ssh 凭证（也可用 CLI 参数） |

### config 字段（运行时用）

```yaml
platform:
  verifier: {impl: remote, url: "http://127.0.0.1:8765", token: "..."}
  benchmarker: {impl: remote, url: "...", token: "..."}
  profiler: {impl: remote, url: "...", token: "..."}
  verification_worker: {impl: remote, url: "...", token: "..."}
  # 不碰 GPU 的组件（specs/roofline/bottleneck/rag）留默认 nvidia/noop，跑在本机
```

### daemon 端点

| 端点 | 用途 |
|---|---|
| `GET /health` | 存活 + GPU 名 |
| `GET /specs?device_name=...` | GPU 规格（供本机 `gpu_name` 配置） |
| `POST /run_test_batch` | 跑 kernel+test，返回 success/stdout/stderr |
| `POST /benchmark_batch` | CUDA event 计时，返回 time_ms/ptx_hash |
| `POST /profile_batch` | NCU profiling，返回 metrics dict |

---

## 七、当前局限 / 后续

### 批量接口（步骤 6 现状）

batch 端点（`/benchmark_batch` 等）**形态已就绪**：daemon 能在一次请求里收 N 个 kernel、串行跑（GPU 锁保证单卡串行正确性）、返回 N 个结果。实测 batch4 与单发 4 次耗时相当（~4×单次），因为：

- **单 GPU 必须串行** benchmark/profile（并发会撞 CUDA context，结果不准甚至崩）——这是硬件约束，不是软件可优化的。
- **瓶颈是每项的 subprocess + torch import（~3.5s/次）**，不是 HTTP 往返（~50ms）。批量不省这个。
- worker 是各自独立进程（方案 C），每个只管自己的 1 个 kernel，跨进程聚合要重编排 OptManager。

所以批量在当前架构下**延迟收益接近零**。`Remote*` 客户端发的是单元素批（一次 1 个），因为调用方（orchestrator）是逐候选串行调 `benchmark_kernel`，不是"一轮一次性给一批"。

### 真要省延迟的方向（后续，未做）

**持久 benchmark 子进程**：让 daemon 维护一个长驻子进程，预 import torch/triton，后续 kernel 直接喂它跑，省掉每个 kernel ~3.5s 的 torch import。这才是 batch 真能省的地方。但侵入式（要改 `kernel_subprocess.py` 支持流式多 kernel + 子进程崩溃恢复），且单卡串行下收益仍有限。当前不做，YAGNI。

### 其他局限

- **daemon 单进程**：uvicorn 单 event loop，并发请求被串行（async 函数里同步 subprocess 阻塞 loop）。单卡下这反而正确（GPU 要串行）；多卡时需开多 worker + 按 GPU 分配。
- **NCU 权限**：NCU 需权限，云端 root 可直接跑；非 root 要配 `KERNELAGENT_NCU_USE_SUDO` 或 sudoers。
- **隧道生命周期**：当前隧道由 deploy 进程持有，deploy 退出隧道可能断。长时间跑可改 `ssh -fN` 后台常驻或 systemd。
- **隧道断了怎么办**：Remote* 调用超时降级（返回 inf/None/False），不崩，但结果无效；需重新 deploy 或手动重连隧道。
