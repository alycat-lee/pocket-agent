from __future__ import annotations

from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMFormatError
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry, tool


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))


def _weather_registry(config: AgentConfig) -> ToolRegistry:
    registry = ToolRegistry(config)

    @tool(registry, "查询指定城市天气")
    def get_weather(city: str) -> str:
        mapping = {"北京": "晴，25℃", "上海": "多云，18℃"}
        return f"{city}：{mapping.get(city, '暂无数据')}"

    @tool(registry, "计算数学表达式")
    def calculator(expression: str) -> str:
        return "7"

    return registry


def test_normal_script_three_steps_no_duplicate_real_calls(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=10)
    llm = mock_llm_factory(
        [
            0,
            {"city": "北京"},
            0,
            {"city": "上海"},
            1,
            {"expression": "25-18"},
            -1,
            "北京与上海温差为7℃",
        ]
    )
    result = AgentRunner(config, llm, _weather_registry(config)).run(
        "查询北京和上海的天气并告诉我温差"
    )

    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.answer == "北京与上海温差为7℃"
    assert result.stats["total_steps"] == 3
    assert result.stats["real_calls"] == 3
    assert result.stats["cache_hits"] == 0
    assert result.stats["intercepts"] == 0
    assert result.stats["prompt_tokens_max"] > 0
    assert [s.tool_name for s in result.steps] == [
        "get_weather",
        "get_weather",
        "calculator",
    ]
    assert all(not s.from_cache for s in result.steps)


def test_step_events_carry_authoritative_turn(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=10)
    llm = mock_llm_factory(
        [
            0,
            {"city": "北京"},
            0,
            {"city": "上海"},
            -1,
            "两地对比完毕",
        ]
    )
    bus = _Bus()
    AgentRunner(config, llm, _weather_registry(config), events=bus).run(
        "查询北京和上海天气",
        turn=1,
    )
    step_payloads = [payload for name, payload in bus.events if name.startswith("step.")]
    assert step_payloads
    for payload in step_payloads:
        assert payload.get("turn") == 1
    answer = next(payload for name, payload in bus.events if name == "answer.done")
    assert answer.get("turn") == 1
    assert "usage" not in answer


def test_answer_done_carries_last_usage(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=10)
    llm = mock_llm_factory([-1, "两地对比完毕"])
    llm.last_usage = {
        "prompt_tokens": 24,
        "completion_tokens": 165,
        "total_tokens": 189,
        "cached_tokens": 2,
    }
    bus = _Bus()
    AgentRunner(config, llm, _weather_registry(config), events=bus).run(
        "查询北京天气",
        turn=1,
    )
    answer = next(payload for name, payload in bus.events if name == "answer.done")
    assert answer["usage"] == llm.last_usage
    assert "prompt_tokens_max" in answer


def test_repeat_script_ledger_intercept_then_consecutive_stop(
    mock_llm_factory: type,
) -> None:
    config = AgentConfig(max_steps=10, max_consecutive_intercepts=2)
    llm = mock_llm_factory(
        [
            0,
            {"city": "北京"},
            0,
            {"city": "北京"},
            0,
            {"city": "北京"},
            "基于缓存信息作答",
        ]
    )
    result = AgentRunner(config, llm, _weather_registry(config)).run("查北京天气")

    assert result.stop_reason == StopReason.CONSECUTIVE_INTERCEPTS
    assert result.answer == "基于缓存信息作答"
    assert result.stats["real_calls"] == 1
    assert result.stats["cache_hits"] == 2
    assert result.stats["intercepts"] == 2
    assert result.stats["total_steps"] == 3
    assert result.steps[0].from_cache is False
    assert result.steps[1].from_cache is True
    assert result.steps[2].from_cache is True


def test_runaway_script_max_steps_graceful_with_answer(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=3, max_consecutive_intercepts=10)
    llm = mock_llm_factory(
        [
            0,
            {"city": "北京"},
            0,
            {"city": "上海"},
            0,
            {"city": "广州"},
            "步数用尽，已收集部分天气信息",
        ]
    )
    result = AgentRunner(config, llm, _weather_registry(config)).run("不停查天气")

    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.answer == "步数用尽，已收集部分天气信息"
    assert result.stats["total_steps"] == 3
    assert result.stats["real_calls"] == 3
    assert isinstance(result.answer, str) and len(result.answer) > 0


def test_exception_script_tool_error_continues_and_recorded(
    mock_llm_factory: type,
) -> None:
    config = AgentConfig(max_steps=10)
    registry = ToolRegistry(config)

    @tool(registry, "总会失败的工具")
    def boom(x: str) -> str:
        raise RuntimeError("爆炸了")

    @tool(registry, "查询指定城市天气")
    def get_weather(city: str) -> str:
        return f"{city}：晴，25℃"

    llm = mock_llm_factory(
        [
            0,
            {"x": "1"},
            1,
            {"city": "北京"},
            -1,
            "天气晴，先前工具出错已忽略",
        ]
    )
    result = AgentRunner(config, llm, registry).run("先失败再查天气")

    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.stats["total_steps"] == 2
    assert result.steps[0].error is not None
    assert "[工具错误]" in result.steps[0].result
    assert "爆炸了" in result.steps[0].result
    assert result.steps[1].tool_name == "get_weather"
    assert result.steps[1].error is None
    assert "晴" in result.steps[1].result
    assert result.answer == "天气晴，先前工具出错已忽略"


def test_weather_choose_prompt_excludes_wiki_and_web(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=4)
    registry = ToolRegistry(config)

    @tool(registry, "查询指定城市天气", domain="weather")
    def get_weather(city: str) -> str:
        return f"{city}：晴，25℃"

    @tool(registry, "获取当前时间", domain="time")
    def get_current_time() -> str:
        return "12:00:00"

    @tool(registry, "查维基百科摘要", domain="knowledge")
    def wiki_lookup(query: str) -> str:
        return "百科"

    @tool(registry, "搜索网页标题摘要", domain="web")
    def web_search(query: str) -> str:
        return "网页"

    llm = mock_llm_factory([0, {"city": "北京"}, -1, "北京晴 25℃"])
    AgentRunner(config, llm, registry).run("查北京、上海、广州天气，算最高最低温差")
    choose = next(p for p in llm.prompts if "可用工具" in p or "工具：" in p)
    assert "wiki" not in choose
    assert "web_search" not in choose
    assert "get_weather" in choose


def test_three_city_weather_not_duplicate_by_args(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=10)
    registry = ToolRegistry(config)

    @tool(registry, "查询指定城市天气", domain="weather")
    def get_weather(city: str) -> str:
        return f"{city}：晴"

    llm = mock_llm_factory(
        [
            0,
            {"city": "北京"},
            0,
            {"city": "上海"},
            0,
            {"city": "广州"},
            -1,
            "三地温差已计算",
        ]
    )
    result = AgentRunner(config, llm, registry).run(
        "查北京、上海、广州天气，算最高最低温差"
    )
    weather = [s for s in result.steps if s.tool_name == "get_weather"]
    assert len(weather) == 3
    assert result.stats["real_calls"] == 3
    assert all(not s.from_cache for s in weather)


def test_choose_format_error_retries_without_step(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=5)
    registry = ToolRegistry(config)

    @tool(registry, "查询指定城市天气", domain="weather")
    def get_weather(city: str) -> str:
        return f"{city}：晴，25℃"

    llm = mock_llm_factory(
        [
            LLMFormatError("bad int"),
            0,
            {"city": "北京"},
            -1,
            "北京晴 25℃",
        ]
    )
    result = AgentRunner(config, llm, registry).run("查北京天气")
    assert all(s.tool_name != "(format_error)" for s in result.steps)
    assert result.steps[0].tool_name == "get_weather"
