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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="运行真机集成测试（需要 llama-server）",
    )
    parser.addoption(
        "--stress",
        action="store_true",
        default=False,
        help="运行专项压测（Phase 9）",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: 需要 llama-server 的真机集成测试")
    config.addinivalue_line("markers", "stress: 专项压测，默认跳过")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="需要 --integration 且 llama-server 可用")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    if not config.getoption("--stress"):
        skip_stress = pytest.mark.skip(reason="需要 --stress")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)


@pytest.fixture
def mock_llm_factory() -> type[MockLLM]:
    return MockLLM
