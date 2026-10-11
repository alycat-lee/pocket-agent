from __future__ import annotations

from typing import ClassVar

from pocket_agent.prompt_registry import PromptRegistry, bind_registry, current_registry
from pocket_agent.prompts import builtin_entries
from pocket_agent.runtime import Context


class PromptsPlugin:
    """内置原子任务提示词 → prompt_registry。"""

    name = "prompts"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["prompt_registry"]

    def apply(self, ctx: Context, config: dict) -> None:
        del config
        registry = PromptRegistry()
        for entry in builtin_entries():
            disposer = registry.register(
                str(entry["id"]),
                variants=dict(entry["variants"]),
                version=str(entry["version"]),
                template=str(entry["template"]),
            )
            ctx.on_dispose(disposer)
        bind_registry(registry)
        ctx.provide("prompt_registry", registry)

        def unbind() -> None:
            if current_registry() is registry:
                bind_registry(None)

        ctx.on_dispose(unbind)
