from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

import httpx
from pocket_agent.config import AgentConfig

_INT_RE = re.compile(r"-?\d+")
_RETRY_DELAYS = (1.0, 2.0)
_MAX_ATTEMPTS = 3  # 首次 + 重试 2 次


class LLMClientProtocol(Protocol):
    """所有 LLM 交互的唯一抽象。测试中注入 MockLLM 实现此协议。"""

    def generate(self, prompt: str) -> str:
        """自由生成（用于 final_answer、summarize）。"""
        ...

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        """约束整数生成（用于 choose_tool）。
        优先用 json_schema 约束；服务不支持时退化为正则提取 + 范围校验。"""
        ...

    def generate_json(self, prompt: str, schema: dict) -> dict:
        """约束 JSON 生成（用于 fill_args）。
        必须保证返回值符合 schema；不符合时抛 LLMFormatError。"""
        ...


class LLMFormatError(Exception):
    """约束解码失效时的兜底异常（理论上不应发生）。"""


class LLMConnectionError(Exception):
    """连接错误、超时或 5xx 在重试耗尽后抛出。"""


class LlamaCppClient:
    """llama.cpp server 的 OpenAI 兼容接口封装，基于 httpx。

    - POST {base_url}/chat/completions
    - generate_int: response_format = {"type": "json_schema",
        "json_schema": {"name": "choice", "schema": {...integer...}}}
      返回 {"value": N} 结构，取 value 字段
    - generate_json: response_format 携带调用方传入的 schema
    - 所有采样参数从 AgentConfig 读取
    - 请求失败（连接错误/超时/5xx）重试 2 次后抛 LLMConnectionError
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def generate(self, prompt: str) -> str:
        """自由生成（用于 final_answer、summarize）。"""
        return self._complete(prompt, response_format=None)

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        """约束整数生成（用于 choose_tool）。"""
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
        return self._parse_int(content, minimum=minimum, maximum=maximum)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        """约束 JSON 生成（用于 fill_args）。"""
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
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._config.temperature,
            "repeat_penalty": self._config.repeat_penalty,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return self._request_with_retry(payload)

    def _request_with_retry(self, payload: dict[str, Any]) -> str:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=self._config.timeout) as client:
                    response = client.post(url, json=payload)
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
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise LLMFormatError(f"unexpected response shape: {response.text!r}") from exc

            if not isinstance(content, str):
                raise LLMFormatError(f"content is not str: {content!r}")
            return content

        raise LLMConnectionError(str(last_error))

    @staticmethod
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
