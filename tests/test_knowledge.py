from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from pocket_agent.config import AgentConfig
from pocket_agent.events import EventStore
from pocket_agent.knowledge import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMPTY_RESULT,
    HashEmbedder,
    KnowledgeBase,
    UnavailableEmbedder,
    attach_sources,
    format_hits,
    parse_document,
    split_text,
)
from pocket_agent.knowledge.embed import LocalEmbedder
from pocket_agent.knowledge.web import is_blocked_url
from pocket_agent.llm_client import LLMFormatError
from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.knowledge_plugin import KnowledgePlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.ui_web import ApprovalBroker, build_app
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context
from pocket_agent.session import SessionManager
from pocket_agent.task import TaskCard, infer_needs_knowledge
from pocket_agent.tools import ToolRegistry
from pocket_agent.verifier import extract_claims, verify

HANDBOOK = Path(__file__).resolve().parent / "fixtures" / "handbook.md"


class _FakeLlm:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.prompts: list[str] = []

    def _next(self) -> Any:
        return self._script.pop(0) if self._script else "ok"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return str(self._next())

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        self.prompts.append(prompt)
        return int(self._next())

    def generate_json(self, prompt: str, schema: dict) -> dict:
        self.prompts.append(prompt)
        value = self._next()
        if not isinstance(value, dict):
            raise LLMFormatError("not a task card")
        return value


class _FakeLlmPlugin:
    name = "llm-fake"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm"]

    def __init__(self, script: list[Any]) -> None:
        self._script = script

    def apply(self, ctx: Context, config: dict) -> None:
        ctx.provide("llm", _FakeLlm(self._script))


class _Bus:
    def emit(self, event: str, payload: dict) -> None:
        del event, payload


def _kb(tmp_path: Path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path / "kb.sqlite", HashEmbedder())


def test_split_text_boundaries_and_overlap() -> None:
    body = "汉" * 520
    chunks = split_text(body, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert CHUNK_SIZE == 500
    assert CHUNK_OVERLAP == 50
    assert len(chunks) == 2
    assert chunks[0] == body[:500]
    assert chunks[1] == body[450:]
    assert chunks[0][-50:] == chunks[1][:50]
    assert split_text("") == []
    assert split_text("短文") == ["短文"]


def test_ingest_then_search_with_hash_embedder(tmp_path: Path) -> None:
    kb = _kb(tmp_path)
    text = HANDBOOK.read_text(encoding="utf-8")
    info = kb.ingest("内部手册.md", text, source="内部手册.md")
    assert info["chunks"] >= 1
    hits = kb.search("星核协议是哪一年签署的")
    assert hits
    assert "2019" in hits[0].text
    assert hits[0].source == "内部手册.md"
    packed = kb.search_text("雨巷灯火的作者")
    assert "[来源：内部手册.md]" in packed
    assert "林小满" in packed


def test_parse_document_txt() -> None:
    assert "星核" in parse_document("note.txt", "星核协议".encode())


def test_citation_positive_and_negative() -> None:
    card = TaskCard(
        type="qa",
        goal="星核协议",
        needs_knowledge=True,
    )
    evidence = "[来源：内部手册.md] 《星核协议》于2019年在北湾签署"
    assert verify("星核协议签署于2019年。", card, evidence=evidence) == []
    bad = verify(
        "星核协议签署于1999年，预算300亿。",
        card,
        evidence=evidence,
    )
    assert "请仅基于检索资料回答" in bad


def test_citation_empty_evidence_requires_honest_phrase() -> None:
    card = TaskCard(type="qa", goal="陷阱", needs_knowledge=True)
    assert verify("未找到相关资料", card, evidence=EMPTY_RESULT) == []
    assert "请仅基于检索资料回答" in verify(
        "紫晶河水母币市值约42亿美元。",
        card,
        evidence=EMPTY_RESULT,
    )


def test_attach_sources_appends_list() -> None:
    text = attach_sources("星核协议签署于2019年。", ["内部手册.md"])
    assert "来源：" in text
    assert "- 内部手册.md" in text


def test_extract_claims_numbers_and_proper_nouns() -> None:
    claims = extract_claims("《星核协议》于2019年签署，预算300亿。")
    assert "2019年" in claims or "2019" in claims
    assert "300" in claims
    assert "星核协议" in claims


def test_hallucination_trap_returns_not_found(
    mock_llm_factory: type,
) -> None:
    cfg = AgentConfig(
        max_steps=3,
        verifier_enabled=True,
        planner_enabled=True,
        answer_max_retries=0,
    )
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "查询紫晶河水母币市值",
                "constraints": {},
                "plan": ["检索", "作答"],
                "needs_history": False,
                "needs_knowledge": True,
            },
            "会话标题",
        ]
    )
    mgr = SessionManager(
        cfg,
        llm,
        AgentRunner(cfg, llm, ToolRegistry(cfg)),
        events=_Bus(),
        knowledge=None,
    )
    answer = mgr.chat("s1", "根据文档介绍紫晶河水母币目前市值多少？")
    assert "未找到相关资料" in answer
    assert "亿美元" not in answer
    assert "不该被调用的编造答案" not in answer


def test_qa_hit_uses_evidence_and_sources(
    tmp_path: Path,
    mock_llm_factory: type,
) -> None:
    kb = _kb(tmp_path)
    kb.ingest("内部手册.md", HANDBOOK.read_text(encoding="utf-8"), source="内部手册.md")
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, planner_enabled=True)
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "星核协议签署年份",
                "constraints": {},
                "plan": ["检索", "作答"],
                "needs_history": False,
                "needs_knowledge": True,
            },
            "星核协议签署于2019年。",
            "协议",
        ]
    )
    mgr = SessionManager(
        cfg,
        llm,
        AgentRunner(cfg, llm, ToolRegistry(cfg)),
        events=_Bus(),
        knowledge=kb,
    )
    answer = mgr.chat("s1", "《星核协议》是哪一年签署的？")
    assert "2019" in answer
    assert "内部手册.md" in answer
    assert any("未找到相关资料" in p or "检索资料" in p for p in llm.prompts[1:])


def test_missing_embedding_dependency_degrades(tmp_path: Path) -> None:
    reason = "未安装 sentence-transformers。请执行 pip install 'pocket-agent[knowledge]'"
    kb = KnowledgeBase(tmp_path / "off.sqlite", UnavailableEmbedder(reason))
    assert kb.available is False
    packed = kb.search_text("星核协议")
    assert packed.startswith(EMPTY_RESULT)
    assert "sentence-transformers" in packed
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        kb.ingest("a.md", "hello")
    local = LocalEmbedder()
    if not local.available:
        assert "sentence-transformers" in local.reason
    else:
        assert local.reason == ""


def test_knowledge_plugin_registers_tools_and_unmounts(tmp_path: Path) -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    ctx.mount(
        KnowledgePlugin(),
        {"embedder": "hash", "db_path": str(tmp_path / "p.sqlite")},
    )
    tools: ToolRegistry = ctx.require("tools")
    names = {t.name for t in tools.all()}
    assert {"search_knowledge", "wiki_lookup", "web_search"} <= names
    search = tools.get("search_knowledge")
    assert search.cacheable is True
    assert len(search.short_desc) <= 20
    kb = ctx.require("knowledge")
    kb.ingest("手册.md", "星核协议于2019年签署", source="手册.md")
    result = tools.get("search_knowledge").fn(query="星核协议")
    assert "[来源：手册.md]" in result
    ctx.unmount("knowledge-base")
    assert not ctx.has("knowledge")
    left = {t.name for t in tools.all()}
    assert "search_knowledge" not in left
    assert "wiki_lookup" not in left
    assert "web_search" not in left


def test_plugin_degrades_when_embedder_unavailable(tmp_path: Path) -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    ctx.mount(
        KnowledgePlugin(),
        {
            "embedder": "unavailable",
            "reason": "未安装 sentence-transformers",
            "db_path": str(tmp_path / "u.sqlite"),
        },
    )
    kb = ctx.require("knowledge")
    assert kb.available is False
    text = ctx.require("tools").get("search_knowledge").fn(query="x")
    assert EMPTY_RESULT in text
    assert "sentence-transformers" in text


def test_infer_needs_knowledge() -> None:
    assert infer_needs_knowledge("紫晶河水母币目前市值多少？") is True
    assert infer_needs_knowledge("北冥星港条约的签署国有哪些？") is True
    assert infer_needs_knowledge("你好") is False


def test_source_format_and_ssrf_block() -> None:
    packed = format_hits([])
    assert packed == EMPTY_RESULT
    assert is_blocked_url("http://169.254.169.254/latest/meta-data")
    assert is_blocked_url("file:///etc/passwd")
    assert not is_blocked_url("https://zh.wikipedia.org/wiki/Test")


def test_knowledge_crud_api(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    ctx = Context()
    ctx.mount(
        _FakeLlmPlugin(
            [
                {
                    "type": "qa",
                    "goal": "x",
                    "constraints": {},
                    "plan": ["a"],
                    "needs_history": False,
                },
                "hi",
            ]
        )
    )
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 3})
    ctx.mount(
        KnowledgePlugin(),
        {"embedder": "hash", "db_path": str(tmp_path / "api.sqlite")},
    )
    ctx.mount(SessionPlugin())
    app = build_app(
        ctx,
        broker=ApprovalBroker(2),
        event_store=EventStore(tmp_path / "events"),
        settings_path=tmp_path / "settings.json",
    )

    async def _call(method: str, path: str, **kwargs: Any) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.request(method, path, **kwargs)

    res = asyncio.run(_call("GET", "/api/knowledge/status"))
    assert res.status_code == 200
    assert res.json()["available"] is True

    res = asyncio.run(
        _call(
            "POST",
            "/api/knowledge/documents",
            json={"name": "手册.md", "text": HANDBOOK.read_text(encoding="utf-8")},
        )
    )
    assert res.status_code == 200, res.text
    doc_id = res.json()["id"]

    res = asyncio.run(_call("GET", "/api/knowledge/documents"))
    assert any(d["id"] == doc_id for d in res.json()["documents"])

    res = asyncio.run(
        _call("POST", "/api/knowledge/search", json={"query": "星核协议哪年签署"})
    )
    body = res.json()
    assert "[来源：" in body["text"]
    assert "2019" in body["text"]

    res = asyncio.run(_call("DELETE", f"/api/knowledge/documents/{doc_id}"))
    assert res.json()["ok"] is True
    res = asyncio.run(_call("GET", "/api/knowledge/documents"))
    assert res.json()["documents"] == []
