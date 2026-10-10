from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
from pocket_agent.llm_client import LLMClientProtocol, LLMConnectionError, LLMFormatError
from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context, PluginError

_INT_RE = re.compile(r"-?\d+")
_RETRY_DELAYS = (1.0, 2.0)
_MAX_ATTEMPTS = 3

# 复杂生成走云端；其余（choose/fill/summarize）走本地
_CLOUD_MARKERS = (
    "基于以上信息回答任务",  # FINAL_ANSWER
    "你正在逐段撰写文章",  # SECTION_WRITE
    "生成一篇约",  # OUTLINE
)


class DeepSeekClient:
    """DeepSeek OpenAI 兼容 API（httpx），实现 LLMClientProtocol。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._timeout = timeout

    def generate(self, prompt: str) -> str:
        return self._complete(prompt, response_format=None)

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "minimum": minimum,
                    "maximum": maximum,
                }
            },
            "required": ["value"],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "choice", "schema": schema},
        }
        content = self._complete(prompt, response_format=response_format)
        return _parse_int(content, minimum=minimum, maximum=maximum)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "tool_args", "schema": schema},
        }
        content = self._complete(prompt, response_format=response_format)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMFormatError(f"invalid json: {content!r}") from exc
        if not isinstance(data, dict):
            raise LLMFormatError(f"expected json object, got {type(data).__name__}")
        return data

    def _complete(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None,
    ) -> str:
        base = self._base_url
        url = (
            f"{base}/chat/completions"
            if base.endswith("/v1")
            else f"{base}/v1/chat/completions"
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < len(_RETRY_DELAYS):
                    time.sleep(_RETRY_DELAYS[attempt])
                    continue
                raise LLMConnectionError(str(exc)) from exc
            if response.status_code >= 500:
                last_error = LLMConnectionError(
                    f"server error {response.status_code}: {response.text}"
                )
                if attempt < len(_RETRY_DELAYS):
                    time.sleep(_RETRY_DELAYS[attempt])
                    continue
                raise last_error
            if response.status_code >= 400:
                raise LLMConnectionError(
                    f"HTTP {response.status_code}: {response.text}"
                )
            try:
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content")
                if not content:
                    alt = message.get("reasoning_content")
                    if isinstance(alt, str) and alt.strip():
                        content = alt
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise LLMFormatError(
                    f"unexpected response shape: {response.text!r}"
                ) from exc
            if not isinstance(content, str):
                raise LLMFormatError(f"content is not str: {content!r}")
            return content
        raise LLMConnectionError(str(last_error))


class RoutingLLM:
    """简单任务走 local，复杂生成走 cloud。"""

    def __init__(self, local: LLMClientProtocol, cloud: LLMClientProtocol) -> None:
        self._local = local
        self._cloud = cloud

    def _pick(self, prompt: str) -> LLMClientProtocol:
        if any(marker in prompt for marker in _CLOUD_MARKERS):
            return self._cloud
        return self._local

    def generate(self, prompt: str) -> str:
        return self._pick(prompt).generate(prompt)

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return self._pick(prompt).generate_int(prompt, minimum, maximum)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return self._pick(prompt).generate_json(prompt, schema)


class LlmDeepSeekPlugin:
    """DeepSeek 云端插件；route=true 时与 llm_local 组成路由并提供 llm。"""

    name = "llm-deepseek"
    requires: list[str]
    provides: list[str]

    def __init__(self) -> None:
        self.requires = []
        self.provides = ["llm"]

    def prepare(self, config: dict) -> None:
        if bool(config.get("route", False)):
            self.requires = ["llm_local"]
            self.provides = ["llm"]
        else:
            self.requires = []
            self.provides = ["llm"]

    def apply(self, ctx: Context, config: dict) -> None:
        api_key_env = str(config.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = str(config.get("api_key") or os.environ.get(api_key_env, ""))
        if not api_key:
            raise PluginError(
                f"llm-deepseek missing API key (env {api_key_env} or config.api_key)"
            )

        ac = agent_config_from(config)
        cloud = DeepSeekClient(
            api_key=api_key,
            base_url=str(config.get("base_url") or "https://api.deepseek.com/v1"),
            model=str(config.get("model") or "deepseek-chat"),
            temperature=ac.temperature,
            top_p=ac.top_p,
            max_tokens=ac.max_tokens,
            timeout=ac.timeout,
        )

        if bool(config.get("route", False)):
            if not ctx.has("llm_local"):
                raise PluginError("llm-deepseek route=true requires llm_local")
            if ctx.has("llm"):
                raise PluginError(
                    "llm already registered; use llm-llamacpp local_only: true with route"
                )
            local = ctx.require("llm_local")
            ctx.provide("llm", RoutingLLM(local, cloud))
            return

        if ctx.has("llm"):
            raise PluginError(
                "llm already registered (互斥：请禁用 llm-llamacpp 或改用 route)"
            )
        ctx.provide("llm", cloud)


def _parse_int(content: str, *, minimum: int, maximum: int) -> int:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "value" in data:
            value = int(data["value"])
            if minimum <= value <= maximum:
                return value
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = _INT_RE.search(content)
    if match is not None:
        value = int(match.group(0))
        if minimum <= value <= maximum:
            return value
    raise LLMFormatError(
        f"cannot parse int in [{minimum}, {maximum}] from {content!r}"
    )
