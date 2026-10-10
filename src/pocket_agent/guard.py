from __future__ import annotations

from dataclasses import dataclass

from pocket_agent.config import AgentConfig


@dataclass
class GuardVerdict:
    should_stop: bool
    reason: str = ""  # "max_steps" | "consecutive_intercepts" | "repeat_output"


class LoopGuard:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._consecutive_intercepts = 0
        self._prev_raw_output: str | None = None

    def on_step(
        self,
        step: int,
        raw_llm_output: str,
        intercepted: bool,
    ) -> GuardVerdict:
        """每步结束后调用。判定规则见文档 02 §3.5。"""
        if intercepted:
            self._consecutive_intercepts += 1
        else:
            self._consecutive_intercepts = 0

        if self._consecutive_intercepts >= self._config.max_consecutive_intercepts:
            self._prev_raw_output = raw_llm_output
            return GuardVerdict(should_stop=True, reason="consecutive_intercepts")

        # 缓存拦截由连续拦截计数负责；重复输出检测仅针对非拦截步
        if (
            not intercepted
            and self._prev_raw_output is not None
            and raw_llm_output == self._prev_raw_output
        ):
            self._prev_raw_output = raw_llm_output
            return GuardVerdict(should_stop=True, reason="repeat_output")

        if step >= self._config.max_steps:
            self._prev_raw_output = raw_llm_output
            return GuardVerdict(should_stop=True, reason="max_steps")

        self._prev_raw_output = raw_llm_output
        return GuardVerdict(should_stop=False, reason="")
