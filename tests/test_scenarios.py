from __future__ import annotations

from pocket_agent.config import AgentConfig
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import LLMFormatError
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry, tool


def test_long_result_compressed(mock_llm_factory: type) -> None:
    config = AgentConfig()
    registry = ToolRegistry(config)

    @tool(registry, "生成数据报告")
    def report_tool(path: str) -> str:
        return "数据报告。" * 1000

    llm = mock_llm_factory(
        [
            0,
            {"path": "report.txt"},
            -1,
            "回答",
        ]
    )
    result = AgentRunner(config, llm, registry).run("读取报告并总结")

    assert result.stop_reason == StopReason.MODEL_DONE
    assert len(result.steps) == 1
    assert len(result.steps[0].result) <= config.per_result_limit
    assert "[...截断" in result.steps[0].result
    # prompt_tokens_max 为粗估 token 数，应对齐 ≈3500 token 预算
    assert result.stats["prompt_tokens_max"] <= 3500
    assert result.answer == "回答"


def test_six_step_chain(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=10)
    registry = ToolRegistry(config)

    @tool(registry, "步进工具")
    def step_tool(n: int) -> str:
        return f"ok-{n}"

    responses: list[object] = []
    for i in range(6):
        responses.extend([0, {"n": i}])
    responses.extend([-1, "六步完成"])

    llm = mock_llm_factory(responses)
    result = AgentRunner(config, llm, registry).run("连续执行六步")

    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.stats["real_calls"] == 6
    # stats 使用 cache_hits，对应 CallLedger.hit_count
    assert result.stats["cache_hits"] == 0

    ledger = CallLedger()
    fingerprints = [
        ledger.fingerprint(step.tool_name, step.args) for step in result.steps
    ]
    assert len(fingerprints) == 6
    assert len(set(fingerprints)) == 6
    assert result.answer == "六步完成"


def test_out_of_range_choice(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=10, max_consecutive_intercepts=2)
    registry = ToolRegistry(config)

    @tool(registry, "查询天气")
    def get_weather(city: str) -> str:
        return f"{city}：晴"

    llm = mock_llm_factory(
        [
            LLMFormatError("constraint decode failed"),
            LLMFormatError("still invalid"),
            0,
            {"city": "北京"},
            -1,
            "北京晴",
        ]
    )

    result = AgentRunner(config, llm, registry).run("查北京天气")

    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.stats["intercepts"] >= 1
    assert any(
        s.tool_name == "(format_error)" and s.error is not None for s in result.steps
    )
    assert any("[格式错误]" in s.result for s in result.steps)
    assert any(s.tool_name == "get_weather" for s in result.steps)
    assert result.answer == "北京晴"
