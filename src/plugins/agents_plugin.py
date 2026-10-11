"""agents 服务插件：目录脚手架 + 存量会话迁移。"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pocket_agent.agents import AgentService
from pocket_agent.runtime import Context


class AgentsPlugin:
    """Agent 目录体系 → agents。"""

    name = "agents"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["agents"]

    def apply(self, ctx: Context, config: dict) -> None:
        data = dict(config or {})
        root = Path(str(data.get("agents_root") or data.get("root") or "agents"))
        legacy = data.get("legacy_sessions")
        service = AgentService(
            root,
            legacy_sessions=Path(str(legacy)) if legacy else None,
        )
        self._register_skills(ctx, service)
        ctx.provide("agents", service)

    def _register_skills(self, ctx: Context, service: AgentService) -> None:
        if not ctx.has("prompt_registry"):
            return
        registry = ctx.require("prompt_registry")
        for profile in service.list():
            for skill in service.list_skills(profile.name):
                local = skill.body.strip()
                if len(local) > 150:
                    local = local[:149] + "…"
                disposer = registry.register(
                    f"skill:{profile.name}:{skill.id}",
                    variants={"cloud": skill.body, "local": local},
                    version="1.0.0",
                    template=skill.body,
                )
                ctx.on_dispose(disposer)
