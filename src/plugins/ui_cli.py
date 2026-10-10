from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

from pocket_agent.runtime import Context

logger = logging.getLogger(__name__)


class UiCliPlugin:
    """CLI 侧事件订阅（无对外服务；装配后保证 agent 可用）。"""

    name = "ui-cli"
    requires: ClassVar[list[str]] = ["agent"]
    provides: ClassVar[list[str]] = []

    def apply(self, ctx: Context, config: dict) -> None:
        verbose = bool(config.get("verbose", False))
        if not verbose:
            return

        def _on_tool(payload: dict[str, Any]) -> None:
            args_text = json.dumps(
                payload.get("args", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            logger.info(
                "[cli] step=%s tool=%s args=%s",
                payload.get("step"),
                payload.get("tool"),
                args_text,
            )

        def _on_cache(payload: dict[str, Any]) -> None:
            logger.info(
                "[cli] cache_hit step=%s tool=%s",
                payload.get("step"),
                payload.get("tool"),
            )

        def _on_intercept(payload: dict[str, Any]) -> None:
            logger.info(
                "[cli] intercepted step=%s reason=%s",
                payload.get("step"),
                payload.get("reason"),
            )

        def _on_answer(payload: dict[str, Any]) -> None:
            logger.info(
                "[cli] answer.done stop=%s steps=%s",
                payload.get("stop_reason"),
                payload.get("total_steps"),
            )

        ctx.on("step.tool_call", _on_tool)
        ctx.on("step.cache_hit", _on_cache)
        ctx.on("step.intercepted", _on_intercept)
        ctx.on("answer.done", _on_answer)
