---
name: anthropic-compatible-gateway-integration
description: |
  Integrate the anthropic Python SDK with a third-party Anthropic-compatible
  gateway (e.g. a cloud proxy fronting DeepSeek/GLM, or DeepSeek's own
  /anthropic endpoint). Use when: (1) AttributeError 'ThinkingBlock' object
  has no attribute 'text' when calling messages.create, (2) HTTP 403
  "has not activated the model X[1m]" / "Please activate the model", (3)
  wiring ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL to reach a bearer-token
  gateway, (4) base_url concatenating into a doubled path like
  /v1/chat/completions/v1/messages, (5) deciding whether a gateway speaks
  Anthropic Messages or OpenAI Chat Completions protocol. Covers: SDK
  self-reading of auth-token/base_url env, base_url truncation, model-name
  suffix stripping, and ThinkingBlock handling for reasoning models.
author: Claude Code
version: 1.0.0
date: 2026-07-23
---

# Anthropic-Compatible Gateway Integration

## Problem

You want to point the `anthropic` Python SDK at a third-party gateway
(e.g. a cloud proxy such as KingCloud `kspmas.ksyun.com` fronting
DeepSeek-V4-Pro, or DeepSeek's own `https://api.deepseek.com/anthropic`)
instead of `api.anthropic.com`. The gateway authenticates with a bearer
token and exposes a base URL that may look like an OpenAI path. Naive
wiring fails in four non-obvious ways, each with a misleading symptom.

## Context / Trigger Conditions

Reach for this skill when you see any of:

- `AttributeError: 'ThinkingBlock' object has no attribute 'text'` (often
  surfaced through pydantic's `__getattr__`) right after a successful
  `messages.create` call.
- HTTP `403 Forbidden` with body like
  `{"error":{"message":"Your account ... has not activated the model deepseek-v4-pro[1m]. Please activate the model ..."}}`.
- A working Claude Code config uses `ANTHROPIC_BASE_URL=https://host/v1/chat/completions`, but your own anthropic-SDK code 404s or routes oddly.
- You have `ANTHROPIC_AUTH_TOKEN` set but the provider's `is_available()` returns False because it only reads `ANTHROPIC_API_KEY`.
- httpx logs show a doubled path: `POST https://host/v1/chat/completions/v1/messages`.

## Solution

### 0. Probe the protocol BEFORE wiring code (don't assume)

A gateway named with `ANTHROPIC_*` env vars is not guaranteed to speak
Anthropic Messages. Probe with a raw POST (no SDK) using a no-proxy opener,
trying both protocols and a couple of base_url truncations:

```python
import os, json, urllib.request
TOKEN = os.environ["ANTHROPIC_AUTH_TOKEN"]
HOST = "https://kspmas.ksyun.com"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # avoid socks/ALL_PROXY
def probe(url, proto, model):
    body = {"model": model, "max_tokens": 8192,
            "messages": [{"role":"user","content":"Reply with exactly: PONG"}]}
    headers = {"Content-Type":"application/json",
               **({"x-api-key":TOKEN,"anthropic-version":"2023-06-01"} if proto=="anthropic"
                  else {"Authorization":f"Bearer {TOKEN}"})}
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    with opener.open(req, timeout=40) as r: print(r.status, r.read().decode()[:300])
# anthropic endpoint
probe(f"{HOST}/v1/messages", "anthropic", "deepseek-v4-pro")
# openai endpoint
probe(f"{HOST}/v1/chat/completions", "openai", "deepseek-v4-pro")
```

A **403 "model not activated"** means the endpoint exists and authed
(routed fine) — it's a model-name problem (see #3), not a routing 404.
A **200** tells you the working protocol + response shape. Anthropic
responses look like `{"content":[{"type":"text","text":"PONG"}], "usage":{...}}`;
OpenAI like `{"choices":[{"message":{"content":"PONG"}}]}`.

### 1. Let the SDK self-read env (don't pass base_url yourself)

The `anthropic` SDK (verified on 0.117.1) reads these env vars itself when
no constructor args are given:

- `ANTHROPIC_AUTH_TOKEN` → sent as `Authorization: Bearer <token>` (this is
  what bearer-token gateways expect; `ANTHROPIC_API_KEY` instead sends
  `x-api-key`).
- `ANTHROPIC_BASE_URL` → the client base URL. Importantly, **even when you
  pass `api_key=` explicitly, `base_url` is still read from env.** So the
  minimal change is: if `ANTHROPIC_API_KEY` is unset but
  `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_BASE_URL` are, construct `Anthropic()`
  with NO args and let the SDK resolve both from env.

```python
api_key = self._get_api_key("ANTHROPIC_API_KEY")
if api_key:
    self.client = Anthropic(api_key=api_key)            # standard path unchanged
elif os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_BASE_URL"):
    self.client = Anthropic()                            # SDK self-reads env
```

Confirm: `client.auth_headers == {'Authorization': 'Bearer <token>'}` and
`client.base_url` is your gateway. No code change to base_url passing
needed.

### 2. base_url = gateway ROOT, not the full chat path

Pass `ANTHROPIC_BASE_URL` as the host root (e.g. `https://kspmas.ksyun.com`),
**not** `https://host/v1/chat/completions`. The SDK appends `/v1/messages`
itself. A trailing `/v1/chat/completions` concatenates into
`/v1/chat/completions/v1/messages` — a doubled path. Some gateways happen
to tolerate it (return 200 anyway), but it's fragile; prefer the clean root.

Note: a Claude Code config may use the full `/v1/chat/completions` base_url
because the Claude Code client normalizes it internally. That does NOT mean
the raw `anthropic` SDK wants the same string.

### 3. Strip model-name suffixes like `[1m]`

Some clients (Claude Code) append context-window markers such as
`deepseek-v4-pro[1m]` and strip them before sending. Third-party gateways
generally do NOT recognize these suffixes and return
`403 ... has not activated the model deepseek-v4-pro[1m]`. Register and call
with the **bare model id** (`deepseek-v4-pro`); long-context is handled by
the gateway, not encoded in the name. If your provider layer passes the
model name through verbatim (most do), do the stripping at registration time.

### 4. Handle ThinkingBlock from reasoning models

Reasoning models (e.g. DeepSeek-V4-Pro, DeepSeek-R1) return
`response.content = [ThinkingBlock, TextBlock]` — the thinking block comes
FIRST and has NO `.text` attribute. The common pattern
`response.content[0].text` raises
`AttributeError: 'ThinkingBlock' object has no attribute 'text'`. Iterate
and take the first `text` block:

```python
@staticmethod
def _extract_text(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":   # skip ThinkingBlock
            return block.text
    return str(response.content)  # fallback; avoid IndexError
```

Use `block.type == "text"` rather than importing `TextBlock`/`ThinkingBlock`
classes — string comparison is robust to SDK internal class renames.

Caveat: with a too-small `max_tokens` (e.g. 16) the model may not finish
thinking, so `content` is ALL thinking blocks and no text block. Use a
realistic `max_tokens` (e.g. 8192) and check `stop_reason == "end_turn"`.

## Verification

```python
import os
os.environ.pop("ANTHROPIC_API_KEY", None)   # force token path
import anthropic
c = anthropic.Anthropic()
assert c.base_url == os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
r = c.messages.create(model="deepseek-v4-pro", max_tokens=8192, temperature=0.0,
                      messages=[{"role":"user","content":"Reply with exactly: PONG"}])
assert r.stop_reason == "end_turn"
text = next(b.text for b in r.content if b.type == "text")
assert text.strip() == "PONG"
```

## Example

Integrating a `BaseProvider` subclass that already calls
`Anthropic(api_key=...)`. Two edits make it gateway-ready:

1. In client init, add the env-self-read fallback (Solution #1).
2. In response parsing, replace `response.content[0].text` with the
   `_extract_text` helper (Solution #4).
3. Register the bare model id (`deepseek-v4-pro`, no `[1m]`) in the model
   registry; point users at `.env`:
   ```
   ANTHROPIC_AUTH_TOKEN=<token>
   ANTHROPIC_BASE_URL=https://kspmas.ksyun.com
   OPENAI_MODEL=deepseek-v4-pro
   ```

## Notes

- `ANTHROPIC_API_KEY` takes precedence if set — it sends `x-api-key` and
  targets `api.anthropic.com` by default. To use a gateway, ensure
  `ANTHROPIC_API_KEY` is UNSET and only `ANTHROPIC_AUTH_TOKEN` +
  `ANTHROPIC_BASE_URL` are set (or the SDK still reads `ANTHROPIC_BASE_URL`
  even with `ANTHROPIC_API_KEY`, but auth header differs — probe to be sure).
- Avoid the `ALL_PROXY=socks://...` trap (httpx rejects socks scheme);
  probe/wraps should use a no-proxy opener or `unset ALL_PROXY all_proxy`.
- `temperature` is honored by most gateways for non-reasoning models; some
  reasoning models pin their own sampling and ignore it — harmless.
- This is about gateway plumbing, not model quality. Kernel/search scaffolds
  around the LLM still determine the ceiling.

## References

- anthropic Python SDK: `Anthropic.__init__` accepts `api_key` and
  `auth_token` (both keyword-only); env resolution for `ANTHROPIC_AUTH_TOKEN`
  and `ANTHROPIC_BASE_URL` confirmed empirically on SDK 0.117.1 — verify
  against your installed version's source (`inspect.getsource`).
- DeepSeek Anthropic-compatible endpoint: `https://api.deepseek.com/anthropic`.
- Reasoning/thinking blocks: Anthropic Messages API `thinking` extended-
  thinking blocks; third-party reasoning models mirror the `content[].type`
  shape.
