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

# Reasoning model returns empty answer (thinking ate the budget)

## Problem

You call a reasoning model (one that emits a `thinking` block before the
final answer) via the Anthropic Messages API (or an Anthropic-compatible
gateway). For **complex** prompts it returns an empty / missing final
answer — `response.content` is `[ThinkingBlock(...)]` with no `TextBlock`.
Simple prompts work fine. The caller gets `""` or `len=0` and no usable
output.

Raising `max_tokens` does not reliably fix it; the model just thinks
longer. Setting `reasoning_effort=max` (where supported) makes it WORSE —
deeper thinking consumes more budget, the answer gets squeezed out harder.

## Context / Trigger Conditions

- Anthropic Python SDK (`client.messages.create`) or an
  Anthropic-compatible gateway (e.g. a cloud proxy fronting DeepSeek).
- Model is a reasoning/thinking model: DeepSeek-R1/V3/V4-Pro, Claude w/
  extended thinking, QwQ, GLM-Zero, etc.
- Response `content` has type `thinking` but no type `text`.
- Symptom scales with prompt complexity: "output f(x)=x+1" works;
  "optimize this 200-line Triton kernel" returns empty.
- Your caller stringified the content list and saw `ThinkingBlock(...)`
  text (a draft, NOT the answer) — common bug where a fallback returns
  the thinking draft as if it were the answer.

## Root Cause

Reasoning models spend tokens on the `thinking` block first, then emit
the final `text` answer. Thinking has no fixed cap by default, so on hard
prompts it runs until it hits `max_tokens` — and the answer is never
produced. The budget is shared: more thinking = less (or no) answer.

This is NOT the model failing to reason — it's reasoning *too long* and
running out of room to answer. Increasing `max_tokens` just gives it more
room to think; increasing `reasoning_effort` explicitly asks for *more*
thinking. Both move the wrong way.

## Solution

**Cap the thinking budget** so the model must stop thinking and answer
within the remaining `max_tokens`. Use the `thinking` parameter:

```python
response = client.messages.create(
    model="deepseek-v4-pro",
    max_tokens=20000,                      # MUST be > budget_tokens
    thinking={"type": "enabled", "budget_tokens": 4096},   # the fix
    messages=[{"role": "user", "content": complex_prompt}],
)
```

Rules:
- `max_tokens` must be **greater than** `budget_tokens` (the SDK raises
  otherwise). A safe ratio: `max_tokens >= budget + 8000` for code answers.
- Start at `budget_tokens=4096` for code-gen tasks; tune up if answers get
  shallow, down if still empty.
- Verified on Anthropic SDK 0.117 + KingCloud gateway fronting
  DeepSeek-V4-Pro: `budget_tokens=1024` → model thinks ~111 chars then
  answers; `budget_tokens=4096` → complex Triton prompts produce complete,
  closed code blocks.

### If the gateway ignores `thinking`

Some gateways don't forward it. Detect by inspecting `response.content`:
if it's still all-thinking with the budget set low, the param isn't
passing through. Fallbacks (in order of preference):
1. Retry with a higher `max_tokens` (gives a thin sliver of answer room);
   output is stochastic so a few retries sometimes land an answer.
2. Shorten/simplify the prompt so thinking doesn't blow up (the model
   over-thinks verbose prompts).
3. Use a non-reasoning model for the task.

## Verification

```python
import anthropic
c = anthropic.Anthropic()  # env: ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL
r = c.messages.create(model="deepseek-v4-pro", max_tokens=8000,
    thinking={"type":"enabled","budget_tokens":1024},
    messages=[{"role":"user","content":"Output f(x)=x+1 in a ```python block. No prose."}])
types = [b.type for b in r.content]
assert "text" in types and any("```python" in b.text for b in r.content if b.type=="text")
# before the fix: types == ['thinking'], no usable text
```

## Example

KernelAgent optimizing Triton kernels: DeepSeek-V4-Pro returned `len=0`
(thinking-only) on a voxelization-optimization prompt ~80% of the time.
Setting `ANTHROPIC_THINKING_BUDGET=4096` (wired into the provider as a
`thinking={"type":"enabled","budget_tokens":int(env)}` kwarg) made the
model reliably emit complete code blocks. Without it, 3 optimization
rounds produced zero usable kernels; with it, the same run optimized
voxelization 0.584ms to 0.456ms (+22%).

A provider-level wiring pattern (env-driven, backward-compatible):

```python
budget = os.getenv("ANTHROPIC_THINKING_BUDGET")
create_kwargs = {"model": model, "max_tokens": max_tokens}
if budget:
    b = int(budget)
    if max_tokens <= b:                # SDK requires max_tokens > budget
        max_tokens = b + max(4096, b); create_kwargs["max_tokens"] = max_tokens
    create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": b}
response = client.messages.create(**create_kwargs)
```

Also fix the `_extract_text` helper: when there's **no** text block, return
`""` — NOT `str(response.content)`. Stringifying a `ThinkingBlock` hands
the caller the thinking *draft* (which may contain half-written code),
masquerading as the answer. Returning empty lets callers retry cleanly.

## Notes

- Do NOT raise `reasoning_effort` / `max_tokens` as the first fix for
  empty answers — that's the trap. Cap thinking first.
- A small budget can hurt answer quality on truly hard problems (the
  model can't reason enough). Tune up, but rarely above ~16k.
- This applies to any reasoning model exposed through the Anthropic
  Messages schema; OpenAI-style `reasoning_effort` is a different knob
  (low/medium/high) — for those, lower the effort, don't raise it.
- If your caller logs "no code block / len=0" intermittently on a
  reasoning model, suspect this immediately.

## References

- Anthropic extended thinking: `thinking={"type":"enabled",
  "budget_tokens": N}`; `max_tokens` must exceed `budget_tokens`.
- Confirmed empirically: Anthropic SDK 0.117 via KingCloud gateway to
  DeepSeek-V4-Pro, budget 1024/4096 produce text answers; unbounded
  thinking on complex prompts returns thinking-only.
- Related: anthropic-compatible-gateway-integration skill (gateway
  plumbing); this skill is the "model never answers" follow-on.
