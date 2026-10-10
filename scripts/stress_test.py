#!/usr/bin/env python3
"""专项压测：长结果压缩、长链条、窗口边界指引。"""

from __future__ import annotations

import time

from pocket_agent.compressor import ContextCompressor
from pocket_agent.config import AgentConfig
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry


class MockLLM:
    def __init__(self, responses: list[object]) -> None:
        self._queue = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._pop()  # type: ignore[return-value]

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        self.prompts.append(prompt)
        return self._pop()  # type: ignore[return-value]

    def generate_json(self, prompt: str, schema: dict) -> dict:
        self.prompts.append(prompt)
        return self._pop()  # type: ignore[return-value]

    def _pop(self) -> object:
        if not self._queue:
            raise AssertionError("MockLLM response queue exhausted")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def stress_long_result() -> list[dict[str, float | int | str]]:
    """长结果压测：1 万 / 5 万字，记录压缩耗时并跑完整 Runner。"""
    rows: list[dict[str, float | int | str]] = []
    for size in (10_000, 50_000):
        config = AgentConfig(max_steps=5)
        compressor = ContextCompressor(config)
        payload = ("压测数据。" * ((size // 5) + 1))[:size]

        t0 = time.perf_counter()
        compressed = compressor.compress_result(payload)
        compress_ms = (time.perf_counter() - t0) * 1000

        registry = ToolRegistry(config)

        def big_report(path: str, *, _body: str = payload) -> str:
            return _body

        big_report.__name__ = "big_report"
        registry.register(big_report, short_desc="超长报告")

        llm = MockLLM([0, {"path": "big.txt"}, -1, "摘要完成"])
        t1 = time.perf_counter()
        result = AgentRunner(config, llm, registry).run(f"总结 {size} 字报告")
        run_ms = (time.perf_counter() - t1) * 1000

        assert result.stats["prompt_tokens_max"] <= 3500, result.stats
        assert len(compressed) <= config.per_result_limit
        if size == 50_000:
            assert compress_ms < 2000, f"5 万字压缩过慢: {compress_ms:.1f}ms"

        rows.append(
            {
                "case": f"long_result_{size}",
                "chars": size,
                "compress_ms": round(compress_ms, 3),
                "run_ms": round(run_ms, 3),
                "prompt_tokens_max": result.stats["prompt_tokens_max"],
                "result_len": len(result.steps[0].result) if result.steps else 0,
            }
        )
    return rows


def stress_long_chain() -> dict[str, float | int | str]:
    """长链条：10 步不同工具，max_steps=15，状态摘要 ≤ 500。"""
    # 关闭召回截断，保证 10 个工具全部进入候选（编号 0..9）
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
    responses.extend([-1, "十步完成"])

    llm = MockLLM(responses)
    result = AgentRunner(config, llm, registry).run("依次调用全部十个工具")

    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.stats["total_steps"] == 10
    assert result.stats["prompt_tokens_max"] <= 3500

    state = ContextCompressor(config).summarize_state(
        [f"{s.step}. {s.tool_name} → {s.result}" for s in result.steps]
    )
    assert len(state) <= 500

    return {
        "case": "long_chain_10",
        "steps": result.stats["total_steps"],
        "prompt_tokens_max": result.stats["prompt_tokens_max"],
        "state_chars": len(state),
        "stop_reason": result.stop_reason.value,
    }


def _print_table(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    header = " | ".join(f"{k:>{widths[k]}}" for k in keys)
    print(header)
    print("-+-".join("-" * widths[k] for k in keys))
    for row in rows:
        print(" | ".join(f"{row[k]!s:>{widths[k]}}" for k in keys))


def print_window_boundary_guide() -> None:
    print()
    print("=== 窗口边界验证（真机，手动）===")
    print("1. 用较小上下文重启 llama-server：")
    print(
        "   llama-server -m <model.gguf> -c 4096 --flash-attn "
        "--cache-reuse 256 --port 8080"
    )
    print("2. 再跑集成测试：")
    print("   py -3.13 -m pytest tests/test_integration.py -v --integration")
    print("3. 对比 -c 8192 的 eval 结果，")
    print("   success_rate 下降即反向验证上下文压缩机制的价值")


def main() -> int:
    print("== stress_long_result ==")
    long_rows = stress_long_result()
    _print_table(long_rows)

    t10 = float(long_rows[0]["compress_ms"])
    t50 = float(long_rows[1]["compress_ms"])
    ratio = t50 / t10 if t10 > 0 else float("inf")
    print(f"compress_ms ratio (50k/10k) = {ratio:.2f}x (期望近似线性、无数量级爆炸)")

    print()
    print("== stress_long_chain ==")
    chain = stress_long_chain()
    _print_table([chain])

    print_window_boundary_guide()
    print()
    print("stress_test finished OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
