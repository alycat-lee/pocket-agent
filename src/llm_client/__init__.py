from __future__ import annotations

import ipaddress
import json
import re
import threading
import time
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import httpx
from pocket_agent.config import AgentConfig

_INT_RE = re.compile(r"-?\d+")
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think\b[^>]*>.*", re.DOTALL | re.IGNORECASE)
_RETRY_DELAYS = (1.0, 2.0)
_MAX_ATTEMPTS = 3  # 首次 + 重试 2 次


def strip_think_blocks(text: str) -> str:
    """移除完整与未闭合的 <think> 块，避免思维链落入回答。"""
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    return cleaned.strip()


def extract_thinking(message: dict[str, Any]) -> str:
    """从 message 收集思维链（think 标签 + reasoning_content），不作为回答。"""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        for match in _THINK_BLOCK.finditer(content):
            inner = re.sub(r"</?think\b[^>]*>", "", match.group(0), flags=re.IGNORECASE)
            if inner.strip():
                parts.append(inner.strip())
        unclosed = _THINK_UNCLOSED.search(_THINK_BLOCK.sub("", content))
        if unclosed is not None:
            inner = re.sub(r"<think\b[^>]*>", "", unclosed.group(0), flags=re.IGNORECASE)
            if inner.strip():
                parts.append(inner.strip())
    alt = message.get("reasoning_content")
    if isinstance(alt, str) and alt.strip():
        parts.append(alt.strip())
    return "\n".join(parts)


def take_thinking(llm: object) -> str:
    """取出并清空客户端上一次请求捕获的思维链。"""
    text = getattr(llm, "last_thinking", "") or ""
    if hasattr(llm, "last_thinking"):
        llm.last_thinking = ""
    return str(text)


def apply_max_tokens(llm: object, max_tokens: int) -> None:
    """尽力把单次生成 max_tokens 写到客户端（不改协议签名）。"""
    if hasattr(llm, "_config") and hasattr(llm._config, "max_tokens"):
        llm._config.max_tokens = max_tokens
    if hasattr(llm, "_max_tokens"):
        llm._max_tokens = max_tokens
    for name in ("_local", "_cloud", "_llm"):
        child = getattr(llm, name, None)
        if child is not None and child is not llm:
            apply_max_tokens(child, max_tokens)
    clients = getattr(llm, "_clients", None)
    if isinstance(clients, dict):
        for child in clients.values():
            if child is not None and child is not llm:
                apply_max_tokens(child, max_tokens)


class LLMClientProtocol(Protocol):
    """所有 LLM 交互的唯一抽象。测试中注入符合本协议的替身实现。"""

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


class LLMAbortedError(Exception):
    """用户中断在途请求；不重试，走优雅收尾。"""


ThinkingMode = Literal["off", "strip", "keep"]


def is_private_host(host: str) -> bool:
    """localhost / 回环 / RFC1918 / 链路本地 → True。"""
    name = (host or "").strip().lower().rstrip(".")
    if name in {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}:
        return True
    if name.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(name)
    except ValueError:
        return False
    return bool(addr.is_private or addr.is_loopback or addr.is_link_local)


def resolve_trust_env(base_url: str, requested: bool | None = None) -> bool:
    """内网/localhost 强制 False；公网才尊重 requested（默认 True）。"""
    parsed = urlparse(base_url if "://" in (base_url or "") else f"http://{base_url}")
    host = parsed.hostname or ""
    if is_private_host(host):
        return False
    return True if requested is None else bool(requested)


class OpenAICompatibleClient:
    """唯一 OpenAI 兼容实现：trust_env / thinking / api_key 由调用方驱动。"""

    def __init__(
        self,
        config: AgentConfig,
        *,
        trust_env: bool | None = None,
        thinking: ThinkingMode | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config = config
        self._api_key = (api_key if api_key is not None else config.api_key) or ""
        if thinking is not None:
            self._thinking: ThinkingMode = thinking
        else:
            self._thinking = "keep" if config.reasoning_enabled else "off"
        self._trust_env = resolve_trust_env(config.base_url, trust_env)
        self.last_thinking: str = ""
        self._abort = threading.Event()
        self._in_flight_lock = threading.Lock()
        self._in_flight: httpx.Response | None = None

    def abort_current(self) -> None:
        """关闭在途 stream，使读取立即失败。"""
        self._abort.set()
        with self._in_flight_lock:
            response = self._in_flight
        if response is not None:
            try:
                response.close()
            except Exception:  # noqa: BLE001, S110 — 中断路径忽略二次关闭
                pass

    def reset_abort(self) -> None:
        self._abort.clear()

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
        if self._thinking == "off":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif self._thinking == "keep":
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if response_format is not None:
            payload["response_format"] = response_format
        return self._request_with_retry(payload)

    def _request_with_retry(self, payload: dict[str, Any]) -> str:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(_MAX_ATTEMPTS):
            if self._abort.is_set():
                raise LLMAbortedError("aborted")
            try:
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
                with (
                    httpx.Client(
                        timeout=self._config.timeout, trust_env=self._trust_env
                    ) as client,
                    client.stream("POST", url, json=payload, headers=headers) as response,
                ):
                    with self._in_flight_lock:
                        self._in_flight = response
                    try:
                        if self._abort.is_set():
                            raise LLMAbortedError("aborted")
                        raw_bytes = response.read()
                    finally:
                        with self._in_flight_lock:
                            self._in_flight = None
            except LLMAbortedError:
                raise
            except httpx.RequestError as exc:
                if self._abort.is_set():
                    raise LLMAbortedError(str(exc)) from exc
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
                body = raw_bytes.decode("utf-8")
                data = json.loads(body)
                message = data["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise LLMFormatError("message is not object")
                thinking = extract_thinking(message)
                self.last_thinking = thinking if self._thinking == "keep" else ""
                raw = message.get("content")
                content = strip_think_blocks(raw) if isinstance(raw, str) else ""
            except LLMFormatError:
                raise
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LLMFormatError(
                    f"unexpected response shape: {raw_bytes[:400]!r}"
                ) from exc

            if not content:
                raise LLMFormatError(
                    "empty content after stripping think blocks "
                    "(reasoning_content is not used as the answer)"
                )
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


class LlamaCppClient(OpenAICompatibleClient):
    """docs/03 兼容旧名：本地默认 thinking=off（reasoning_enabled 时 keep）。"""

    def __init__(self, config: AgentConfig) -> None:
        thinking: ThinkingMode = "keep" if config.reasoning_enabled else "off"
        super().__init__(config, trust_env=None, thinking=thinking)
