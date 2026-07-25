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

# 接 Anthropic 兼容网关

## 问题

你想把 `anthropic` Python SDK 指向第三方网关（如代理 DeepSeek-V4-Pro 的金山云 `kspmas.ksyun.com`，或 DeepSeek 自家的 `https://api.deepseek.com/anthropic`），而非 `api.anthropic.com`。网关用 bearer token 鉴权，base URL 可能长得像 OpenAI 路径。朴素接线会在 4 个非显而易见处失败，每个症状都误导人。

## 触发条件

看到以下任一就用本 skill：

- `AttributeError: 'ThinkingBlock' object has no attribute 'text'`（常经 pydantic `__getattr__` 浮出），紧接一次成功的 `messages.create` 之后。
- HTTP `403 Forbidden`，body 形如 `{"error":{"message":"Your account ... has not activated the model deepseek-v4-pro[1m]. Please activate the model ..."}}`。
- 可用的 Claude Code 配置用 `ANTHROPIC_BASE_URL=https://host/v1/chat/completions`，但你自己写的 anthropic-SDK 代码 404 或路由怪。
- 设了 `ANTHROPIC_AUTH_TOKEN` 但 provider 的 `is_available()` 返回 False，因为它只读 `ANTHROPIC_API_KEY`。
- httpx 日志显示双路径：`POST https://host/v1/chat/completions/v1/messages`。

## 解决

### 0. 接线代码前先探测协议（别假设）

用 `ANTHROPIC_*` env 命名的网关不一定说 Anthropic Messages 协议。用裸 POST（不经 SDK）+ no-proxy opener 探测，两种协议 + 几种 base_url 截断都试：

```python
import os, json, urllib.request
TOKEN = os.environ["ANTHROPIC_AUTH_TOKEN"]
HOST = "https://kspmas.ksyun.com"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 避 socks/ALL_PROXY
def probe(url, proto, model):
    body = {"model": model, "max_tokens": 8192,
            "messages": [{"role":"user","content":"Reply with exactly: PONG"}]}
    headers = {"Content-Type":"application/json",
               **({"x-api-key":TOKEN,"anthropic-version":"2023-06-01"} if proto=="anthropic"
                  else {"Authorization":f"Bearer {TOKEN}"})}
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    with opener.open(req, timeout=40) as r: print(r.status, r.read().decode()[:300])
# anthropic 端点
probe(f"{HOST}/v1/messages", "anthropic", "deepseek-v4-pro")
# openai 端点
probe(f"{HOST}/v1/chat/completions", "openai", "deepseek-v4-pro")
```

**403 "model not activated"** = 端点存在且鉴权通过（路由 OK），是模型名问题（见 #3），不是路由 404。
**200** = 拿到可用协议 + 响应形态。Anthropic 响应形如 `{"content":[{"type":"text","text":"PONG"}], "usage":{...}}`；OpenAI 形如 `{"choices":[{"message":{"content":"PONG"}}]}`。

### 1. 让 SDK 自读 env（别自己传 base_url）

`anthropic` SDK（0.117.1 验证）在不传构造参数时自己读这些 env：

- `ANTHROPIC_AUTH_TOKEN` → 作 `Authorization: Bearer <token>` 发（bearer-token 网关要的就是这个；`ANTHROPIC_API_KEY` 则发 `x-api-key`）。
- `ANTHROPIC_BASE_URL` → client base URL。关键是：**即使你显式传 `api_key=`，`base_url` 仍从 env 读**。所以最小改动：若 `ANTHROPIC_API_KEY` 没设但 `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_BASE_URL` 有，就 `Anthropic()` 不传参，让 SDK 从 env 解析两者。

```python
api_key = self._get_api_key("ANTHROPIC_API_KEY")
if api_key:
    self.client = Anthropic(api_key=api_key)            # 标准路径不变
elif os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_BASE_URL"):
    self.client = Anthropic()                            # SDK 自读 env
```

确认：`client.auth_headers == {'Authorization': 'Bearer <token>'}`，`client.base_url` 是你的网关。不必改 base_url 传参代码。

### 2. base_url 用网关根，不要带完整 chat 路径

`ANTHROPIC_BASE_URL` 设主机根（如 `https://kspmas.ksyun.com`），**不要** `https://host/v1/chat/completions`。SDK 自己拼 `/v1/messages`。带 `/v1/chat/completions` 尾巴会拼成 `/v1/chat/completions/v1/messages`——双路径。有些网关恰好容忍（仍返回 200），但脆弱，用干净根。

注意：Claude Code 配置可能用带 `/v1/chat/completions` 的 base_url，因 Claude Code 客户端内部规整。这不代表裸 `anthropic` SDK 要同样字符串。

### 3. 剥掉模型名后缀如 `[1m]`

有些客户端（Claude Code）加上下文窗口标记如 `deepseek-v4-pro[1m]`，发请求前剥掉。第三方网关一般不认这些后缀，返回 `403 ... has not activated the model deepseek-v4-pro[1m]`。注册和调用都用**裸 model id**（`deepseek-v4-pro`）；长上下文由网关处理，不编码进名。若你的 provider 层原样透传 model 名（多数如此），在注册时剥。

### 4. 处理推理模型的 ThinkingBlock

推理模型（如 DeepSeek-V4-Pro、DeepSeek-R1）返回 `response.content = [ThinkingBlock, TextBlock]`——thinking 块在前，且**没有 `.text` 属性**。常见写法 `response.content[0].text` 抛 `AttributeError: 'ThinkingBlock' object has no attribute 'text'`。遍历取首个 `text` 块：

```python
@staticmethod
def _extract_text(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":   # 跳过 ThinkingBlock
            return block.text
    return ""   # 无 text block（纯 thinking）→ 返回空，让调用方干净重试，
                # 别返回 str(response.content)——会把思考草稿当答案
```

用 `block.type == "text"` 而非 import `TextBlock`/`ThinkingBlock` 类——字符串比较不受 SDK 内部类改名影响。

注意：`max_tokens` 太小（如 16）时模型可能没思考完，`content` 全是 thinking 块无 text。用合理 `max_tokens`（如 8192）+ 查 `stop_reason == "end_turn"`。若模型常因思考吃光预算不答，见 `reasoning-model-thinking-budget` skill。

## 验证

```python
import os
os.environ.pop("ANTHROPIC_API_KEY", None)   # 强制走 token 路径
import anthropic
c = anthropic.Anthropic()
assert c.base_url == os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
r = c.messages.create(model="deepseek-v4-pro", max_tokens=8192, temperature=0.0,
                      messages=[{"role":"user","content":"Reply with exactly: PONG"}])
assert r.stop_reason == "end_turn"
text = next(b.text for b in r.content if b.type == "text")
assert text.strip() == "PONG"
```

## 例子

把一个已调 `Anthropic(api_key=...)` 的 `BaseProvider` 子类改成支持网关，两处编辑：

1. client init 加 env 自读 fallback（解决 #1）。
2. 响应解析把 `response.content[0].text` 换成 `_extract_text` helper（解决 #4）。
3. 模型注册表注册裸 model id（`deepseek-v4-pro`，无 `[1m]`）；`.env` 指引：
   ```
   ANTHROPIC_AUTH_TOKEN=<token>
   ANTHROPIC_BASE_URL=https://kspmas.ksyun.com
   OPENAI_MODEL=deepseek-v4-pro
   ```

## 注意

- `ANTHROPIC_API_KEY` 设了优先——它发 `x-api-key`、默认指向 `api.anthropic.com`。要用网关，确保 `ANTHROPIC_API_KEY` 未设、只设 `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`（或 SDK 即使设了 `ANTHROPIC_API_KEY` 也读 `ANTHROPIC_BASE_URL`，但 auth header 不同——探测确认）。
- 避 `ALL_PROXY=socks://...` 坑（httpx 拒 socks scheme）；探测/包装用 no-proxy opener 或 `unset ALL_PROXY all_proxy`。
- `temperature` 多数网关对非推理模型生效；部分推理模型自定采样忽略它——无害。
- 这是网关管道，不是模型质量。LLM 周边的 kernel/搜索脚手架仍决定上限。

## 参考

- anthropic Python SDK：`Anthropic.__init__` 接 `api_key` 和 `auth_token`（均 keyword-only）；`ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL` 的 env 解析在 SDK 0.117.1 实测确认——以你装版本源码为准（`inspect.getsource`）。
- DeepSeek Anthropic 兼容端点：`https://api.deepseek.com/anthropic`。
- reasoning/thinking 块：Anthropic Messages API `thinking` 扩展思考块；第三方推理模型镜像 `content[].type` 形态。
