---
name: reasoning-model-thinking-budget
description: |
  Fix reasoning/thinking models (DeepSeek-R1/V3/V4, Claude with extended
  thinking, QwQ, etc.) that return an EMPTY answer — no final text/code,
  only a thinking block — when the task is complex. Use when: (1) an
  Anthropic-API reasoning model's response.content has only a
  `thinking` block and NO `text` block (caller gets "" / len=0), (2) the
  model "thinks forever" and never emits the final answer, (3) you set
  max_tokens but complex prompts still produce empty outputs while simple
  ones work, (4) you're about to raise max_tokens or set effort=max to fix
  this (WRONG — that makes it worse). Covers: why thinking eats the token
  budget, the `thinking.budget_tokens` cap, and the common mistake of
  increasing max_tokens/effort instead.
author: Claude Code
version: 1.0.0
date: 2026-07-24
---

# 推理模型返回空答案（thinking 吃光了预算）

## 问题

你通过 Anthropic Messages API（或 Anthropic 兼容网关）调推理模型（先输出 `thinking` 块、再出最终答案的模型）。**复杂** prompt 时它返回空/没有最终答案——`response.content` 是 `[ThinkingBlock(...)]`，没有 `TextBlock`。简单 prompt 正常。调用方拿到 `""` 或 `len=0`，没可用输出。

调大 `max_tokens` 不一定能修（模型只是想得更久）。设 `reasoning_effort=max`（若支持）会更糟——思考更深消耗更多预算，答案更没空间。

## 触发条件

- Anthropic Python SDK（`client.messages.create`）或 Anthropic 兼容网关（如代理 DeepSeek 的云网关）。
- 模型是推理/思考模型：DeepSeek-R1/V3/V4-Pro、Claude extended thinking、QwQ、GLM-Zero 等。
- 响应 `content` 只有 `thinking` 类型，没有 `text` 类型。
- 症状随 prompt 复杂度上升："输出 f(x)=x+1" 能成；"优化这段 200 行 Triton kernel" 返回空。
- 调用方把 content 列表 stringify 看到 `ThinkingBlock(...)` 文本（是草稿，不是答案）——常见 bug：fallback 把思考草稿当答案返回。

## 根因

推理模型先花 token 在 `thinking` 块上，再输出最终 `text` 答案。思考默认无上限，遇到难 prompt 就一直想到撞 `max_tokens`——答案永远没产出。预算是共享的：思考越多，答案越少（甚至没有）。

这不是模型"不会推理"——是**想太久没空间答**。调大 `max_tokens` 只是给它更多空间想；调 `reasoning_effort` 是明确要它**想更多**。两个都走反方向。

## 解决

**给思考设上限**，让模型必须停下思考、在剩余 `max_tokens` 里答。用 `thinking` 参数：

```python
response = client.messages.create(
    model="deepseek-v4-pro",
    max_tokens=20000,                      # 必须 > budget_tokens
    thinking={"type": "enabled", "budget_tokens": 4096},   # 关键
    messages=[{"role": "user", "content": complex_prompt}],
)
```

规则：
- `max_tokens` 必须 **大于** `budget_tokens`（否则 SDK 报错）。代码任务建议 `max_tokens >= budget + 8000`。
- 代码生成从 `budget_tokens=4096` 起调：答案太浅就调大，仍空就调小。
- 实测 Anthropic SDK 0.117 + 金山云网关代理 DeepSeek-V4-Pro：`budget_tokens=1024` → 模型思考 ~111 字符就答；`budget_tokens=4096` → 复杂 Triton prompt 产出完整闭合代码块。

### 若网关不透传 `thinking`

有些网关不转发。检测：设小 budget 后 `response.content` 仍全是 thinking → 参数没透传。回退方案（按优先级）：
1. 调大 `max_tokens` 重试（给答案留一线空间）；输出有随机性，几次重试有时能撞出答案。
2. 缩短/简化 prompt（推理模型对啰嗦 prompt 想太多）。
3. 任务改用非推理模型。

## 验证

```python
import anthropic
c = anthropic.Anthropic()  # env: ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL
r = c.messages.create(model="deepseek-v4-pro", max_tokens=8000,
    thinking={"type":"enabled","budget_tokens":1024},
    messages=[{"role":"user","content":"Output f(x)=x+1 in a ```python block. No prose."}])
types = [b.type for b in r.content]
assert "text" in types and any("```python" in b.text for b in r.content if b.type=="text")
# 修复前：types == ['thinking']，无可用 text
```

## 例子

KernelAgent 优化 Triton kernel：DeepSeek-V4-Pro 对 voxelization 优化 prompt 约 80% 概率返回 `len=0`（纯 thinking）。设 `ANTHROPIC_THINKING_BUDGET=4096`（在 provider 里接成 `thinking={"type":"enabled","budget_tokens":int(env)}`）后，模型稳定产出完整代码块。没它，3 轮优化产出 0 个可用 kernel；有它，同一次跑把 voxelization 从 0.584ms 优化到 0.456ms。

provider 层接线模式（env 驱动、向后兼容）：

```python
budget = os.getenv("ANTHROPIC_THINKING_BUDGET")
create_kwargs = {"model": model, "max_tokens": max_tokens}
if budget:
    b = int(budget)
    if max_tokens <= b:                # SDK 要求 max_tokens > budget
        max_tokens = b + max(4096, b); create_kwargs["max_tokens"] = max_tokens
    create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": b}
response = client.messages.create(**create_kwargs)
```

同时修 `_extract_text`：当**没有** text block 时返回 `""`——**不要**返回 `str(response.content)`。把 `ThinkingBlock` stringify 会把思考**草稿**（可能含半成品代码）当答案给调用方。返回空让调用方能干净重试。

## 注意

- 别把"调 `reasoning_effort`/`max_tokens`"当空答案的第一修——那是陷阱。先给思考设上限。
- 小 budget 在真难题上会伤答案质量（模型没想够）。往上调，但一般不超 ~16k。
- 适用任何走 Anthropic Messages schema 的推理模型；OpenAI 式 `reasoning_effort` 是另一套（low/medium/high）——那个要**调低** effort，不是调高。
- 若调用方在推理模型上间歇性记 "no code block / len=0"，立即怀疑这个。

## 参考

- Anthropic extended thinking：`thinking={"type":"enabled", "budget_tokens": N}`；`max_tokens` 须大于 `budget_tokens`。
- 实测：Anthropic SDK 0.117 经金山云网关到 DeepSeek-V4-Pro，budget 1024/4096 产出 text 答案；复杂 prompt 无 bound 的 thinking 返回纯 thinking。
- 关联：`anthropic-compatible-gateway-integration` skill（网关接线）；本 skill 是"模型不答"的后续。
