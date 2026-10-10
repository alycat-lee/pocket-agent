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
        manager = SessionManager(ac, llm, agent, events=ctx)
        ctx.provide("sessions", manager)
