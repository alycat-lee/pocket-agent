from __future__ import annotations

from typing import ClassVar

from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context


class AgentLoopPlugin:
    """AgentRunner 主循环 → agent（requires: llm, tools）。"""

    name = "agent-loop"
    requires: ClassVar[list[str]] = ["llm", "tools"]
    provides: ClassVar[list[str]] = ["agent"]

    def apply(self, ctx: Context, config: dict) -> None:
        ac = agent_config_from(config)
        llm = ctx.require("llm")
        tools = ctx.require("tools")
        runner = AgentRunner(ac, llm, tools, events=ctx)
        if ctx.has("checkpoint"):
            runner._checkpoints = ctx.require("checkpoint")
        ctx.provide("agent", runner)
