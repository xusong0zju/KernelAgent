# 项目沉淀知识（Skills）

这里收录在 KernelAgent 项目实战中沉淀的、与本项目强相关的非显而易见知识。
每篇对应一个 `~/.claude/skills/<name>/SKILL.md`（工程外，跨项目自动复用）；
本目录是工程内副本，让团队/接手者直接可读，不依赖个人 `~/.claude/`。

## 索引

| 文档 | 何时看 | 一句话 |
|---|---|---|
| [anthropic-compatible-gateway-integration](anthropic-compatible-gateway-integration.md) | 接第三方 Anthropic 兼容网关（金山云 kspmas / DeepSeek 兼容端点） | 模型名别带 `[1m]` 后缀、base_url 用网关根、SDK 自读 `ANTHROPIC_AUTH_TOKEN`、推理模型返回 `[ThinkingBlock,TextBlock]` 要跳 thinking |
| [triton-py312-pyssize-clean-crash](triton-py312-pyssize-clean-crash.md) | Triton kernel 编译报 `PY_SSIZE_T_CLEAN macro must be defined` | triton 3.3 的 C 扩展与 py3.12 不兼容，建 py3.11 环境（ka_gpu）即可 |
| [reasoning-model-thinking-budget](reasoning-model-thinking-budget.md) | 推理模型（deepseek-v4-pro 等）复杂 prompt 返回空（`len=0`，纯 thinking） | 限制 `thinking.budget_tokens`（设 `ANTHROPIC_THINKING_BUDGET`），让思考收敛、给答案留空间——与"effort 调 max"相反 |
| [atomic-scatter-triton-vs-cuda-attribution](atomic-scatter-triton-vs-cuda-attribution.md) | 看到"CUDA 比 Triton 快"想归因"Triton 表达力" | 先做 naive 同结构 A/B（隔离编译器）、random vs sorted（隔离局部性/归约）。真因常是编译器在随机高冲突 atomic 下的优势，不是表达力；输入有序时 Triton 向量化能逼近 CUDA |

## 与项目各文档的关系

- **接 LLM**：`anthropic-compatible-gateway-integration` ← `docs/安装与环境交接.md` §6
- **远程环境**：`triton-py312-pyssize-clean-crash` ← `docs/远程GPU部署.md` 坑1（ka_gpu 来源）
- **LLM 优化闭环**：`reasoning-model-thinking-budget` ← `docs/具身3D算子优化实录.md` 发现1（voxelization 优化靠它）
- **CUDA vs Triton 归因**：`atomic-scatter-triton-vs-cuda-attribution` ← `docs/具身3D算子优化实录.md` 深度对照

## 为什么双份（工程内 + 工程外）

- **工程外**（`~/.claude/skills/`）：被 Claude Code 在任何项目自动命中复用，是"行为级"沉淀——下次类似任务会自动触发建议。
- **工程内**（`docs/skills/`）：团队可读、随仓库走、进 git，是"文档级"沉淀——人不靠 agent 也能查。

两者内容一致；更新时**两边都改**（或以工程内为准定期同步到工程外）。
