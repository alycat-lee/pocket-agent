from __future__ import annotations

from typing import Any

import pytest


class MockLLM:
    """实现 LLMClientProtocol：按预设队列依次返回响应，记录所有 prompt。

    队列元素可以是 str / int / dict，也可以是 BaseException 实例（弹出时抛出）。
    """

    def __init__(self, responses: list[Any]) -> None:
        self._queue: list[Any] = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._pop()

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        self.prompts.append(prompt)
        return self._pop()

    def generate_json(self, prompt: str, schema: dict) -> dict:
        self.prompts.append(prompt)
        return self._pop()

    def _pop(self) -> Any:
        if not self._queue:
            raise AssertionError("MockLLM response queue exhausted")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def mock_llm_factory() -> type[MockLLM]:
    return MockLLM
