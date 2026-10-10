from __future__ import annotations

from typing import ClassVar

from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context
from pocket_agent.session import SessionManager


class SessionPlugin:
    """多轮会话 → sessions。"""

    name = "session"
    requires: ClassVar[list[str]] = ["llm", "agent"]
    provides: ClassVar[list[str]] = ["sessions"]

    def apply(self, ctx: Context, config: dict) -> None:
        ac = agent_config_from(config)
        llm = ctx.require("llm")
        agent = ctx.require("agent")
        writer = ctx.require("writer") if ctx.has("writer") else None
        workspace = ctx.require("workspace") if ctx.has("workspace") else None
        manager = SessionManager(
            ac, llm, agent, events=ctx, writer=writer, workspace=workspace
        )
        ctx.provide("sessions", manager)
