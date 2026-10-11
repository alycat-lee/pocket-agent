from __future__ import annotations

from typing import ClassVar

from pocket_agent.orchestrator import Orchestrator
from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context


class OrchestratorPlugin:
    """TaskCard.plan → 多 worker 编排（requires: agent, llm）。"""

    name = "orchestrator"
    requires: ClassVar[list[str]] = ["agent", "llm"]
    provides: ClassVar[list[str]] = ["orchestrator"]

    def apply(self, ctx: Context, config: dict) -> None:
        ac = agent_config_from(config)
        agent = ctx.require("agent")
        llm = ctx.require("llm")
        orch = Orchestrator(
            ac,
            llm,
            agent,
            agent._registry,
            events=ctx,
        )
        ctx.provide("orchestrator", orch)
