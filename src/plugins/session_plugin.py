from __future__ import annotations

import logging
from typing import ClassVar

from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context
from pocket_agent.session import SessionManager

logger = logging.getLogger(__name__)


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
        orchestrator = ctx.require("orchestrator") if ctx.has("orchestrator") else None
        knowledge = ctx.require("knowledge") if ctx.has("knowledge") else None
        agents = ctx.require("agents") if ctx.has("agents") else None
        sessions_dir = None
        if agents is not None:
            sessions_dir = agents.sessions_dir("default")
        else:
            raw = config.get("sessions_dir")
            if raw:
                sessions_dir = raw
            else:
                logger.warning("会话未持久化")
        manager = SessionManager(
            ac,
            llm,
            agent,
            events=ctx,
            writer=writer,
            workspace=workspace,
            sessions_dir=sessions_dir,
            orchestrator=orchestrator,
            knowledge=knowledge,
            agents=agents,
        )
        ctx.provide("sessions", manager)
