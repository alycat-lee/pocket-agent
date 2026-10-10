from __future__ import annotations

import time

import pytest
from pocket_agent.compressor import ContextCompressor
from pocket_agent.config import AgentConfig
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry


@pytest.mark.stress
def test_stress_long_result(mock_llm_factory: type) -> None:
    timings: list[tuple[int, float]] = []
    for size in (10_000, 50_000):
        config = AgentConfig(max_steps=5)
        payload = ("压测数据。" * ((size // 5) + 1))[:size]
        compressor = ContextCompressor(config)

        t0 = time.perf_counter()
        out = compressor.compress_result(payload)
        elapsed = time.perf_counter() - t0
        timings.append((size, elapsed))

        assert len(out) <= config.per_result_limit
        assert "[...截断" in out

        registry = ToolRegistry(config)

        def big_report(path: str, *, _body: str = payload) -> str:
            return _body

        big_report.__name__ = "big_report"
        registry.register(big_report, short_desc="超长报告")

        llm = mock_llm_factory([0, {"path": "big.txt"}, -1, "ok"])
        result = AgentRunner(config, llm, registry).run(f"总结{size}字")
        assert result.stats["prompt_tokens_max"] <= 3500
        assert result.stop_reason == StopReason.MODEL_DONE

    # 5 万字压缩 < 2 秒
    assert timings[1][1] < 2.0
    # 性能近似线性：50k 不应比 10k 慢超过 20 倍（数量级爆炸即失败）
    if timings[0][1] > 0:
        assert timings[1][1] / timings[0][1] < 20


@pytest.mark.stress
def test_stress_long_chain(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=15, tool_recall_threshold=20)
    registry = ToolRegistry(config)
    for i in range(10):

        def _make(n: int):
            def _fn() -> str:
                return f"step-{n}-done"

            _fn.__name__ = f"tool_{n}"
            return _fn

        registry.register(_make(i), short_desc=f"工具{i}")

    responses: list[object] = []
    for i in range(10):
        responses.extend([i, {}])
    responses.extend([-1, "完成"])

    llm = mock_llm_factory(responses)
    result = AgentRunner(config, llm, registry).run("跑完十条工具链")

    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.stats["total_steps"] == 10
    assert result.stats["prompt_tokens_max"] <= 3500

    state = ContextCompressor(config).summarize_state(
        [f"{s.step}. {s.tool_name} → {s.result}" for s in result.steps]
    )
    assert len(state) <= 500
    assert state.startswith("已完成 10 步：")
