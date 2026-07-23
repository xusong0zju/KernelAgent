# RMSNorm 算子优化实录（千问核心归一化，KernelAgent + 远程 RTX 2080 Ti）

> 本文记录一次真实算子优化：用 KernelAgent 的闭环（本机 LLM 生成 + 云端 GPU 真跑真测）优化千问模型用到的 RMSNorm，并实测优化空间。
> 配套：远程 daemon 部署见 `docs/远程GPU部署.md`，闭环价值论证见 `docs/为何KernelAgent强于直接用LLM.md`。
> 写于 2026-07-23，已端到端跑通。

---

## 一、算子选择与优化空间评估

### 为什么选 RMSNorm

千问（Qwen）/LLaMA 系模型用 **RMSNorm** 替代 LayerNorm 作为归一化层——它是这些模型每个 Transformer block 都跑的热点算子。计算是：`y = x / sqrt(mean(x², dim=-1) + eps)`，沿 hidden 维归约。

它比 ReLU"复杂"在哪：
- 不是纯 elementwise——要先沿 hidden 维**归约**算 sum-of-squares，再**广播**归一化写回。
- **访存密集**：要读一遍 x、写一遍 y，但朴素实现要先读一遍算 rms、再读一遍归一化（**两遍访存**）。
- 访存模式、block size、归约方式都有优化空间——是优化能拉开差距的算子，不是一行搞定的。

### 优化空间量化（动手前）

千问真实 shape：`[4096, 4096]`（batch*seq × hidden，Qwen2-7B 的 hidden=4096），fp32 约 64MB。

| 实现 | 实测 | 说明 |
|---|---|---|
| PyTorch eager | 0.388 ms | 参考基线 |
| 初始 Triton kernel（两遍访存） | 0.368 ms | 我手写的 baseline，已是 eager 的 96% |

**核心优化空间**：初始 kernel 是**两遍遍历**（先算 sum-of-squares，再归一化写回）——访存翻倍。RMSNorm 是 memory-bound，访存是瓶颈，所以**两遍融合成单遍能省一半访存**。带宽下限（读64MB+写64MB≈128MB @ 2080Ti 616GB/s）≈ 0.208ms，初始 0.368ms 离它约 2x 空间。**这就是 KernelAgent 该发现并优化的点。**

---

## 二、闭环怎么跑的

用 `examples/optimize_rmsnorm_remote.py` 驱动，复用 KernelAgent 已验证的组件：

```
本机 deepseek-v4-pro (金山云)          云端 RTX 2080 Ti daemon
  生成优化 kernel ──ssh 隧道──▶ POST /run_test_batch  (验证正确性)
                                  POST /benchmark_batch (CUDA event 计时)
 ◀─── 结构化结果 (success/time_ms/stderr)
  若通过且更快 → 更新 best；若编译/数值失败 → 丢弃，保留 best
```

每轮给 LLM 喂：当前 best kernel + 当前真实时间 + eager 基线 + 优化方向（memory-bound、两遍融合单遍、coalescing、Turing 68 SM 占用率），让它生成优化 kernel；云端验证 + 计时；按真实计时选 best。**3 轮迭代**：

| 轮 | 结果 | 处理 |
|---|---|---|
| 1 | 0.372 ms（没超 baseline 0.368） | 丢弃 |
| 2 | **0.165 ms（+55%）** | ✅ 采纳为新 best |
| 3 | shared-memory 单遍融合，Triton 报 `Unsupported ptr type` 编译失败 | 丢弃，保留 round2 |

### 最终结果（三次复测，误差 <0.3%）

```
initial (两遍):  0.368 ms
best    (LLM优化): 0.165 ms   → 比 initial 快 2.2x (+55%)，比 eager 快 2.4x
eager:           0.388 ms
```

**优化空间评估被实测验证了**——"两遍融合单遍省一半访存"的判断落地为真实的 2.2x 加速。best kernel（`examples/rmsnorm_best.py`）确实把两遍改成了更高效的实现。

---

## 三、闭环价值的实锤

这次最有说服力的是 **round 3**：LLM 生成了一段"看起来对"的 shared-memory 单遍融合代码，但 Triton 不支持 `tl.store(x_smem + pid, x)` 这种 ptr 类型，**编译失败**。

- **直接用 LLM / 通用 agent**：会把这段"看着像优化了、实际编译不过"的代码直接交付给你。你得到的是坏 kernel。
- **KernelAgent 闭环**：云端 GPU 真跑编译，拿到真实 stderr（`Unsupported ptr type`），**客观判定失败，自动丢弃，保留 round2 的真能跑的 best**。

这就是"LLM 负责生成候选、GPU 负责判真假判快慢、闭环负责留好弃坏"——那份分析文档论证的价值，在这里用一次真实优化兑现了。

---

## 四、踩坑记录

### 坑 1：大 shape 在 2080Ti OOM

最初用仓库自带的 `examples/optimize_02_rmsnorm/problem.py`（NCHW `[112,64,512,512]` = 7.6GB），在 22GB 的 2080Ti 上 test 跑 `CUDA out of memory`——光输入就 7.6GB，加参考输出 + kernel 输出 + 中间 tensor 超了。

**解法**：换成千问真实 shape `[4096,4096]`（64MB），既贴合千问实际（RMSNorm 是对 `[batch*seq, hidden]` 沿 hidden 归约，不是 NCHW），又轻松放进 2080Ti。见 `examples/optimize_qwen_rmsnorm/`。

### 坐标坑：带宽下限数字误导

脚本里硬编码的"带宽下限 12.4ms"是按旧 NCHW 7.6GB 算的，没跟着换成 64MB 问题，导致输出里冒出 "best is 7518% of bandwidth limit" 这种荒谬数字。**忽略输出里那行**——真实带宽下限是 ~0.208ms（128MB @ 616GB/s）。best 0.165ms 略低于它，是 L2 缓存/warmup 效应，可复现（三次测量 0.1644/0.1649/0.1648ms）。

### 坑 2：daemon 占显存 + 重复部署端口冲突

`Benchmark.benchmark_pytorch`（eager 基线）是**进程内**直接调 torch.cuda，在 daemon 进程里留下 ~7.6GB 显存占用不释放。重复 deploy 时新 daemon bind 端口失败（`address already in use`），隧道连到旧 daemon → token 401。

**解法**：deploy.py 的 `deploy()` 开头加 `pkill` 旧 daemon（按端口），再 sync/launch（已修）。若仍残留，手动 `pkill -9 -f remote_daemon` + 等 2s 再 deploy。

### 坑 3：NCU profiling 在 2080Ti 没命中

`profile_triton_kernel` 用 `--launch-skip=3`，对 launch 次数少的 kernel 会全跳过（"No kernels were profiled"），且 2080Ti 不在 gpu_specs 表（bottleneck_analyzer 会崩）。所以这次优化是 **benchmark-driven**（真实计时选 best），不是 NCU-driven（瓶颈分析驱动）。NCU 驱动需要修 launch-skip 或加 2080Ti 到 gpu_specs 表，留作后续。

---

## 五、产物

| 文件 | 作用 |
|---|---|
| `examples/optimize_qwen_rmsnorm/problem.py` | 千问 RMSNorm problem（`[4096,4096]`，真实 shape） |
| `examples/optimize_qwen_rmsnorm/input.py` | 初始 Triton kernel（两遍，baseline 0.368ms） |
| `examples/optimize_qwen_rmsnorm/test.py` | 正确性校验（allclose 1e-2） |
| `examples/optimize_rmsnorm_remote.py` | 优化驱动脚本（本机 LLM + 远程 GPU 闭环，可复用跑别的算子） |
| `examples/rmsnorm_best.py` | 优化后 kernel（0.165ms） |

---

## 六、怎么复现

```bash
# 1. 部署远程 daemon（见 docs/远程GPU部署.md）
env -u ALL_PROXY -u all_proxy ~/miniconda3/envs/kernel_agent/bin/python \
  -m triton_kernel_agent.remote.deploy \
  --ssh-host <host> --ssh-port <port> --ssh-user root --ssh-pass <pass> \
  --repo /home/xusong/PreResearch_WS/KernelAgent \
  --port 8765 --python-bin /root/miniconda3/envs/ka_gpu/bin/python
# 记下打印的 token

# 2. 跑优化（本机需配金山云 env: ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL）
env -u ALL_PROXY -u all_proxy OPENAI_MODEL=deepseek-v4-pro \
~/miniconda3/envs/kernel_agent/bin/python examples/optimize_rmsnorm_remote.py \
  --url http://127.0.0.1:8765 --token <上一步token> --rounds 3

# 3. 关机前收尾（关 daemon + 隧道）
~/miniconda3/envs/kernel_agent/bin/python -m triton_kernel_agent.remote.deploy \
  --ssh-host <host> --ssh-port <port> --ssh-user root --ssh-pass <pass> \
  --teardown --port 8765
```

换算子：复制 `optimize_qwen_rmsnorm/` 改 problem/input/test，改脚本的 `OPT_PROMPT` 优化方向，即可用同一闭环优化别的算子。
