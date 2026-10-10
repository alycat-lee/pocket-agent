from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pocket_agent.checkpoint import CheckpointStore
from pocket_agent.runtime import Context


class CheckpointPlugin:
    """断点续跑存储 → checkpoint。"""

    name = "checkpoint"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["checkpoint"]

    def apply(self, ctx: Context, config: dict) -> None:
        root = config.get("root")
        store = CheckpointStore(Path(root) if root is not None else None)
        ctx.provide("checkpoint", store)
