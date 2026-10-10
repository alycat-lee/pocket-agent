from __future__ import annotations

from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry, tool


def _registry(config: AgentConfig) -> ToolRegistry:
    registry = ToolRegistry(config)

    @tool(registry, "查询天气")
    def get_weather(city: str) -> str:
        return f"{city}：晴"

    return registry


class _AlwaysChoose:
    def generate(self, prompt: str) -> str:
        return "final"

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return 0

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return {"city": "北京"}


def test_stop_after_first_tool_call() -> None:
    config = AgentConfig(max_steps=10)
    runner = AgentRunner(config, _AlwaysChoose(), _registry(config))
    done_payloads: list[dict[str, Any]] = []

    class _Bus:
        def emit(self, event: str, payload: dict[str, Any]) -> None:
            if event == "step.tool_call":
                runner.request_stop()
            if event == "answer.done":
                done_payloads.append(payload)

    runner._events = _Bus()
    result = runner.run("查北京天气", session_id="stop1")
    assert result.stop_reason == StopReason.STOPPED
    assert result.stats["total_steps"] >= 1
    assert done_payloads
    assert done_payloads[0]["stop_reason"] == "stopped"
    assert "llm_ms" in done_payloads[0]
    assert "tool_ms" in done_payloads[0]
    assert "cache_hit_rate" in done_payloads[0]
    assert "real_calls" in done_payloads[0]
    assert "intercepts" in done_payloads[0]
    assert "prompt_tokens_max" in done_payloads[0]


def test_step_events_include_status_fields() -> None:
    config = AgentConfig(max_steps=3)
    runner = AgentRunner(config, _AlwaysChoose(), _registry(config))
    seen: dict[str, dict[str, Any]] = {}

    class _Bus:
        def emit(self, event: str, payload: dict[str, Any]) -> None:
            if event.startswith("step.") and event not in seen:
                seen[event] = payload
            if event == "step.tool_call":
                runner.request_stop()

    runner._events = _Bus()
    runner.run("查", session_id="fields")
    for name in ("step.start", "step.tool_call"):
        assert name in seen
        p = seen[name]
        assert "step_index" in p
        assert "elapsed_ms" in p
        assert "prompt_tokens_est" in p
        assert "ttft_ms" in p
        assert p["ttft_ms"] is None
        assert p.get("session_id") == "fields"
