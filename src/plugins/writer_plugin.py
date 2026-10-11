from __future__ import annotations

from typing import ClassVar

from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context
from pocket_agent.writer import LongWriter


class WriterPlugin:
    """长文本生成器 → writer。"""

    name = "writer"
    requires: ClassVar[list[str]] = ["llm"]
    provides: ClassVar[list[str]] = ["writer"]

    def apply(self, ctx: Context, config: dict) -> None:
        ac = agent_config_from(config)
        llm = ctx.require("llm")
        writer = LongWriter(ac, llm)
        ctx.provide("writer", writer)
        # session 只依赖 llm/agent，可能先于 writer 挂载；补绑避免创作任务落入 agent 循环
        if ctx.has("sessions"):
            ctx.require("sessions")._writer = writer
