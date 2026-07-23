# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic provider implementation."""

import os

from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def __init__(self):
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        if not ANTHROPIC_AVAILABLE:
            return

        # Prefer an explicit ANTHROPIC_API_KEY (standard Anthropic / api.anthropic.com).
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()
            # Initialize client (proxy configured via environment variables)
            self.client = Anthropic(api_key=api_key)
            return

        # Fall back to letting the SDK read env itself. This supports
        # Anthropic-compatible gateways that authenticate via a bearer token
        # (ANTHROPIC_AUTH_TOKEN) and a custom ANTHROPIC_BASE_URL, e.g. the
        # KingCloud kspmas gateway fronting DeepSeek-V4-Pro. The SDK resolves
        # both from the environment when no constructor args are passed.
        if os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_BASE_URL"):
            self._original_proxy_env = configure_proxy_environment()
            self.client = Anthropic()

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")

        user_content = messages[-1]["content"] if messages else ""
        response = self.client.messages.create(
            model=model_name,
            max_tokens=min(
                kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
            ),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": user_content}],
        )

        return LLMResponse(
            content=self._extract_text(response), model=model_name, provider=self.name
        )

    @staticmethod
    def _extract_text(response) -> str:
        """Return the first text block from a Messages response.

        Reasoning models (e.g. DeepSeek-V4-Pro) prepend a ``thinking`` block;
        ``response.content[0]`` is then a ``ThinkingBlock`` without a ``.text``
        attribute. Skip non-text blocks and take the first ``TextBlock``.
        """
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        # No text block at all — fall back to a stringified response so the
        # caller still gets something meaningful rather than an IndexError.
        return str(response.content)

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        return [
            self.get_response(
                model_name,
                messages,
                temperature=kwargs.get("temperature", 0.7) + i * 0.1,
            )
            for i in range(n)
        ]

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None

    @property
    def name(self) -> str:
        return "anthropic"
