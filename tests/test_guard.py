from __future__ import annotations

from pocket_agent.config import AgentConfig
from pocket_agent.guard import LoopGuard


def test_no_stop_on_normal_step() -> None:
    guard = LoopGuard(AgentConfig(max_steps=10, max_consecutive_intercepts=2))
    verdict = guard.on_step(step=1, raw_llm_output="choose:0", intercepted=False)
    assert verdict.should_stop is False
    assert verdict.reason == ""


def test_max_steps_reason() -> None:
    guard = LoopGuard(AgentConfig(max_steps=3, max_consecutive_intercepts=2))
    assert guard.on_step(1, "a", False).should_stop is False
    assert guard.on_step(2, "b", False).should_stop is False
    verdict = guard.on_step(3, "c", False)
    assert verdict.should_stop is True
    assert verdict.reason == "max_steps"


def test_consecutive_intercepts_triggers_at_two() -> None:
    guard = LoopGuard(AgentConfig(max_steps=10, max_consecutive_intercepts=2))
    v1 = guard.on_step(1, "out-1", intercepted=True)
    assert v1.should_stop is False
    v2 = guard.on_step(2, "out-2", intercepted=True)
    assert v2.should_stop is True
    assert v2.reason == "consecutive_intercepts"


def test_consecutive_intercepts_reset_after_normal_step() -> None:
    guard = LoopGuard(AgentConfig(max_steps=10, max_consecutive_intercepts=2))
    assert guard.on_step(1, "a", intercepted=True).should_stop is False
    # 中间插入正常步，计数清零
    assert guard.on_step(2, "b", intercepted=False).should_stop is False
    assert guard.on_step(3, "c", intercepted=True).should_stop is False
    # 只有连续第 2 次拦截才终止
    v = guard.on_step(4, "d", intercepted=True)
    assert v.should_stop is True
    assert v.reason == "consecutive_intercepts"


def test_repeat_output_reason() -> None:
    guard = LoopGuard(AgentConfig(max_steps=10, max_consecutive_intercepts=5))
    assert guard.on_step(1, "same", intercepted=False).should_stop is False
    verdict = guard.on_step(2, "same", intercepted=False)
    assert verdict.should_stop is True
    assert verdict.reason == "repeat_output"


def test_priority_consecutive_intercepts_over_repeat_output() -> None:
    """连续拦截优先于重复输出（文档 02 §3.5）。"""
    guard = LoopGuard(AgentConfig(max_steps=10, max_consecutive_intercepts=2))
    guard.on_step(1, "same", intercepted=True)
    verdict = guard.on_step(2, "same", intercepted=True)
    assert verdict.reason == "consecutive_intercepts"


def test_priority_repeat_output_over_max_steps() -> None:
    guard = LoopGuard(AgentConfig(max_steps=2, max_consecutive_intercepts=5))
    guard.on_step(1, "same", intercepted=False)
    verdict = guard.on_step(2, "same", intercepted=False)
    assert verdict.reason == "repeat_output"
