from __future__ import annotations

from typing import ClassVar

from pocket_agent.demo_tools import register_demo_tools
from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context
from pocket_agent.tools import ToolRegistry


class ToolsCorePlugin:
    """内置演示工具 → tools。"""

    name = "tools-core"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["tools"]

    def apply(self, ctx: Context, config: dict) -> None:
        registry = ToolRegistry(agent_config_from(config))
        register_demo_tools(registry)
        ctx.provide("tools", registry)
