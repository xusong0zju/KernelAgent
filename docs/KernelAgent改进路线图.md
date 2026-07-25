# KernelAgent 改进路线图

> 基于本次 RMSNorm / Voxelization / BEV lift-splat 三算子实战暴露的痛点，按"证据强度 × 性价比"排出的改进方向。每条标注**证据来源**（实测/推断），避免把推测当结论。
>
> 写于 2026-07-25。证据见 `docs/具身3D算子优化实录.md`、`docs/用CUDA当探针优化Triton.md`、`docs/RMSNorm优化实录.md`。

---

## 改进的依据（本次实测暴露的 9 个事实）

| # | 事实 | 证据 |
|---|---|---|
| 1 | LLM 生成质量随算子复杂度急剧下降：RMSNorm(纯elementwise)成功，voxelization(atomic)反复失败 | 实测：voxelization 3 轮 0 可用 kernel |
| 2 | 推理模型默认不收敛到代码，必须限 thinking budget | 实测：设 budget 前 80% `len=0`，设后稳定产出 |
| 3 | LLM 反复踩固定几类 Triton 语法坑（continue/tl.static_shared/2D 切片/缺逗号） | 实测：多轮 stderr 集中在这几类 |
| 4 | 闭环到顶后单范式再无突破（Triton 0.456 后 30 轮也到不了 0.370） | 实测：triton_vec 需"切 CUDA 再回 Triton"才出现 |
| 5 | 闭环缺乏瓶颈定位能力（不知是 atomic/归约/编译器哪个主导） | 实测：靠手写 CUDA 拆变量才定位 |
| 6 | 部分算子 Triton 表达力不足（BEV 的向量索引/多列 sort） | 实测：BEV 反复编译失败 |
| 7 | NCU 杀手锏在 2080Ti 失效，只能 benchmark-driven | 实测：NCU 返回 None |
| 8 | 工程稳定性差（token 漂移、首次编译超时、daemon 残留） | 实测：反复 401、30s 超时 |
| 9 | 闭环已能接受 CUDA 候选，但 LLM 不会生成 CUDA | 实测：best.py 是手写 CUDA |

---

## P0：高证据强度 + 高性价比（建议优先做）

### P0-1. Triton 语法陷阱提示已加进 prompt，但可做成"错误自愈"闭环

**问题**：LLM 反复踩 `continue`/`tl.static_shared`/2D 切片/`constexpr` 这几类固定坑（事实 3）。现在靠 prompt 文字提示，但 LLM 仍会忘。

**改进**：在 verify 失败时，**把 stderr 的错误类型映射成针对性补丁提示**，自动加进下一轮 prompt。例如：
- `unsupported AST node type: Continue` → 提示"把 continue 换成 mask"
- `no attribute 'static_shared'` → 提示"用寄存器 + tl.atomic_add，无 shared mem API"
- `unsupported tensor index: constexpr[0]` → 提示"别用向量索引 lin[i]，标量算"
- `arange's arguments must be constexpr` → 提示"PD 标 tl.constexpr"

**性价比**：高。这是把"事实 3 的固定坑"从"碰运气避免"变成"确定性修复"。实现量小（一张错误→提示映射表），收益大（直接救回大量失败轮次）。比让 LLM 自己记住靠谱。

**证据强度**：高（stderr 类型集中、可枚举）。

### P0-2. 瓶颈定位：闭环加"性能探针"阶段（核心架构改进）

**问题**：闭环到顶后不知瓶颈在哪（事实 5），纯调参数无方向。这次靠手写 CUDA 拆变量才看清"向量化 atomic 是主因、归约只值 10%"。

**改进**：闭环在"N 轮无突破"后触发**探针阶段**：
1. 让 LLM 生成多个"只差一个变量"的变体（不同 BLOCK、有无归约、有无向量化），互相对比
2. 跑 random vs sorted 输入（隔离局部性 vs 归约）
3. 跑 naive 同结构 A/B（隔离编译器 vs 算法）
4. 把定位结果（"瓶颈是 X 不是 Y"）灌进下一轮 prompt

**性价比**：高。这是 `docs/用CUDA当探针优化Triton.md` 方法论的工程化。直接针对事实 4+5——把"突破靠人"变成"突破靠闭环"。

**证据强度**：高（这次手做就是这流程，有效）。

### P0-3. 突破单范式：支持 CUDA 候选生成 + 跨语言翻译

**问题**：LLM 困在 Triton 范式（事实 4、9），想不到用 Triton 本就支持但文档不强调的特性（2D 向量化 atomic）。

**改进**：两步——
1. **让 LLM 能生成 CUDA 候选**：现在 `optimize_kernel_remote.py` 已支持 CUDA kernel_code（best.py 能是 CUDA），但 LLM 的 prompt 只教 Triton。加一个"生成 CUDA 探针版"的 prompt 分支，让 LLM 产出 cpp_extension 形式的 CUDA 候选。
2. **跨语言翻译**：定位到 CUDA 的关键技巧后，用一轮"把 CUDA 技巧翻译回 Triton"的 prompt，产 triton_vec 类候选。

**性价比**：中高。针对事实 4 的根因（范式盲区）。实现中等（要处理 CUDA 编译/缓存，已踩过坑可复用）。

**证据强度**：高（这次 triton_vec 就是这流程的产物，1.5x 收益）。

---

## P1：中证据强度 + 中高性价比

### P1-1. 算子复杂度分级 + 自适应策略

**问题**：RMSNorm（简单）和 voxelization（复杂）用同一套 3 轮闭环，但前者轻松成功、后者全失败（事实 1）。

**改进**：闭环前先估计算子复杂度（看 problem.py：有无 atomic/scatter/动态索引/循环依赖），分级：
- **简单**（elementwise/融合）：现有 3 轮够
- **中等**（有 atomic 但结构规整）：加 P0-1 语法自愈 + 增轮次
- **复杂**（scatter/动态 key/多列重排）：直接上 P0-2 探针 + P0-3 CUDA 通道，别在纯 Triton 里空耗

**性价比**：中高。避免对复杂算子做注定失败的"纯 Triton 微调"，省大量无效 LLM 调用。

**证据强度**：中（复杂度分级标准需细化，但"简单 vs 复杂差异巨大"是实测的）。

### P1-2. 修 NCU 在 2080Ti（恢复杀手锏）

**问题**：NCU profiling 返回 None，闭环退化为 benchmark-driven（事实 7），失去 KernelAgent 最核心的优势（见 `docs/为何KernelAgent强于直接用LLM.md`）。

**改进**：两件——
1. 把 2080Ti 加进 `gpu_specs` 表（bottleneck_analyzer 崩溃）
2. 调 `--launch-skip`：低 launch-count kernel 的 launch-skip 要自适应（当前固定值跳过所有 launch）

**性价比**：中。NCU-driven 是 KernelAgent 的"杀手锏"，修复后复杂算子优化质量会上一个台阶（benchmark-driven 看不到 stall 原因）。

**证据强度**：中（NCU 失效已实测，但"修了能恢复多少"待验证）。

### P1-3. thinking budget 已落地，但可自适应

**问题**：`ANTHROPIC_THINKING_BUDGET=4096` 是固定值（事实 2）。简单算子不需要这么多思考预算（浪费），极复杂算子可能不够（答案浅）。

**改进**：按算子复杂度（P1-1 的分级）自适应 budget：简单→1024，中等→4096，复杂→8192。或按"首轮是否产出代码"动态调（产出失败就加预算重试）。

**性价比**：中。小优化，但减少简单算子的 token 浪费 + 复杂算子的浅答。

**证据强度**：中（budget 必要性已证，自适应值是推断）。

---

## P2：工程稳定性（证据强但非核心能力）

### P2-1. 修 daemon token 漂移

**问题**：每次 deploy 后 token 经常对不上（事实 8），反复 401，浪费大量调试时间。根因：pkill 不彻底，旧 daemon 残留 + 新 daemon bind 失败 + health 不校验 token 所以连旧 daemon 也 200。

**改进**：
- deploy 的 pkill 已改进（-9 + 验证杀干净），但可加"deploy 后强制 health 用新 token 校验，失败就报错而非返回看似成功的 token"
- daemon 启动后把 token 写进 health 响应（调试用，加开关）
- 或让 deploy 失败时明确报"端口被占、旧 daemon 未杀"

**性价比**：高（纯省调试时间，不做核心能力也值得）。

### P2-2. CUDA 编译预热 + 持久 subprocess

**问题**：首次 CUDA JIT 编译超 30s test 超时（事实 8）；每个 kernel subprocess 重 import torch ~3.5s。

**改进**：
- CUDA 候选首次编译走"预热"通道（单独 endpoint，编译完缓存），verify 只加载
- 持久 benchmark subprocess（避免每 kernel 重 import torch）——之前文档列为 YAGNI，但 CUDA 路径下编译成本更高，值得重评

**性价比**：中。CUDA 路径下收益变大（编译比 torch import 贵得多）。

---

## P3：探索性（证据弱 / 长期）

### P3-1. 多模型底座 + 模型路由

**问题**：deepseek-v4-pro 在复杂算子上过思考 + 代码质量不足（事实 1+2）。RMSNorm 成功靠它简单。

**改进**：接入更强的代码模型（Claude/gpt-5）做复杂算子，deepseek 做简单算子（省成本）。按 P1-1 复杂度路由模型。

**性价比**：低中。依赖外部模型可用性，但"底座质量决定上限"是事实。

### P3-2. Triton 表达力不足算子的 CUDA fallback

**问题**：BEV 因向量索引/多列 sort 在 Triton 里难表达（事实 6）。

**改进**：闭环检测到"反复同类 Triton 语法错误"时，自动切 CUDA 路径（让 LLM 生成 CUDA 而非继续撞 Triton 墙）。

**性价比**：低中。和 P0-3 协同，但 BEV 类算子占比未知。

### P3-3. reflexion 跨算子迁移

**问题**：每个算子从零学，不积累"向量化 atomic 对 scatter 有效"这类跨算子知识。

**改进**：reflexion 库跨算子复用——voxelization 学到的"2D 向量化 atomic"自动作为 BEV/其他 scatter 算子的初始 prompt 提示。

**性价比**：低。长期价值高，但需要积累多个算子经验才显效。

---

## 推荐执行顺序

1. **P0-1 语法自愈**（1-2 天，立竿见影救失败轮次）
2. **P2-1 token 漂移**（半天，纯省心）
3. **P1-2 修 NCU**（1-2 天，恢复杀手锏）
4. **P0-2 瓶颈定位探针**（3-5 天，核心架构改进，本次方法论工程化）
5. **P0-3 CUDA 候选 + 跨语言翻译**（3-5 天，与 P0-2 协同）
6. **P1-1 复杂度分级**（2 天，串联以上自适应）
7. P1-3 / P2-2 / P3 视进展推进

---

## 与已有沉淀的关系

- **P0-2/P0-3** = `docs/用CUDA当探针优化Triton.md` 方法论的工程化
- **P0-1** = `docs/skills/` 里 Triton 语法坑的自动化（目前只在 prompt 文字提示）
- **P1-2** = `docs/为何KernelAgent强于直接用LLM.md` 里"NCU 是杀手锏"的兑现（当前失效）
- **P1-1** = `docs/具身3D算子优化实录.md` 里"简单 vs 复杂算子差异"的对策
