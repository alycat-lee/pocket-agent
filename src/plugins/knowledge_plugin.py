"""知识库插件：embedding + sqlite + 三个检索工具。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from pocket_agent.knowledge import KnowledgeBase
from pocket_agent.knowledge import web as knowledge_web
from pocket_agent.runtime import Context
from pocket_agent.tools import ToolRegistry

Disposer = Callable[[], None]


def register_knowledge_tools(
    registry: ToolRegistry,
    kb: KnowledgeBase,
    *,
    searxng_url: str = "",
    serper_key: str = "",
) -> list[Disposer]:
    """注册 §3 三个检索工具；描述 ≤20 字且全部 cacheable。"""

    def search_knowledge(query: str) -> str:
        return kb.search_text(query)

    def wiki_lookup(query: str) -> str:
        return knowledge_web.wiki_lookup(query)

    def web_search(query: str) -> str:
        return knowledge_web.web_search(
            query,
            searxng_url=searxng_url,
            serper_key=serper_key,
        )

    return [
        registry.register_with_disposer(
            search_knowledge, "检索私有知识库", domain="knowledge"
        ),
        registry.register_with_disposer(
            wiki_lookup, "查维基百科摘要", domain="knowledge"
        ),
        registry.register_with_disposer(web_search, "搜索网页标题摘要", domain="web"),
    ]


class KnowledgePlugin:
    """私有知识库 + 检索工具 → knowledge。"""

    name = "knowledge-base"
    requires: ClassVar[list[str]] = ["tools"]
    provides: ClassVar[list[str]] = ["knowledge"]

    def apply(self, ctx: Context, config: dict) -> None:
        data = dict(config or {})
        db_path = Path(str(data.get("db_path") or ".knowledge/kb.sqlite"))
        kb = KnowledgeBase.from_config(
            {
                "embedder": data.get("embedder") or "lexical",
                "db_path": str(db_path),
                "top_k": data.get("top_k") or 5,
                "result_limit": data.get("result_limit")
                or data.get("per_result_limit")
                or 300,
                "embed_model": data.get("embed_model") or "",
                "embed_base_url": data.get("embed_base_url") or "",
                "embed_api_key": data.get("embed_api_key") or "",
                "reason": data.get("reason") or "",
            }
        )
        tools: ToolRegistry = ctx.require("tools")
        for disposer in register_knowledge_tools(
            tools,
            kb,
            searxng_url=str(data.get("searxng_url") or ""),
            serper_key=str(data.get("serper_key") or ""),
        ):
            ctx.on_dispose(disposer)
        ctx.provide("knowledge", kb)
