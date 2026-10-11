from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import pytest
from pocket_agent.agents import (
    MEMBERS,
    AgentNameError,
    AgentService,
    validate_agent_name,
)
from pocket_agent.config import AgentConfig
from pocket_agent.plugins.agents_plugin import AgentsPlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context
from pocket_agent.session import SessionManager
from pocket_agent.tools import ToolRegistry


def _svc(tmp_path: Path) -> AgentService:
    return AgentService(tmp_path / "agents", legacy_sessions=tmp_path / "no_legacy")


def test_validate_blocks_path_injection() -> None:
    for bad in ("../etc", "..\\windows", "foo/bar", "foo\\bar", "..", ".hidden", "shared"):
        with pytest.raises(AgentNameError):
            validate_agent_name(bad)
    assert validate_agent_name("写手") == "写手"
    assert validate_agent_name("default") == "default"


def test_scaffold_has_six_members(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    profile = svc.create("写手", description="长文写手")
    assert profile.name == "写手"
    root = svc.agent_dir("写手")
    for member in MEMBERS:
        assert (root / member).exists(), member
    assert (root / "memory" / "entities.json").is_file()
    assert (root / "agent.yml").is_file()
    assert (root / "prompt.md").is_file()
    names = {p.name for p in svc.list()}
    assert {"default", "写手"} <= names


def test_shared_memory_readable_private_wins(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    svc.create("写手")
    svc.write_shared_entities({"公司": "东洲", "城市": "北湾"})
    svc.write_entities("写手", {"公司": "私有工作室"})
    merged_writer = svc.merge_entities("写手")
    merged_default = svc.merge_entities("default")
    assert merged_writer["公司"] == "私有工作室"
    assert merged_writer["城市"] == "北湾"
    assert merged_default["公司"] == "东洲"
    assert merged_default["城市"] == "北湾"


def test_delete_keep_files_and_purge(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    svc.create("临时")
    path = svc.agent_dir("临时")
    svc.delete("临时", keep_files=True)
    assert path.is_dir()
    assert (path / "prompt.md").is_file()
    with pytest.raises(KeyError):
        svc.get("临时")
    assert all(p.name != "临时" for p in svc.list())
    svc.create("干净")
    gone = svc.agent_dir("干净")
    svc.delete("干净", keep_files=False)
    assert not gone.exists()
    with pytest.raises(ValueError):
        svc.delete("default")


def test_legacy_sessions_migrate_idempotent(tmp_path: Path) -> None:
    legacy = tmp_path / ".sessions"
    legacy.mkdir()
    payload = {"session_id": "old1", "title": "旧会话", "turns": []}
    (legacy / "old1.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    svc = AgentService(tmp_path / "agents", legacy_sessions=legacy)
    dest = svc.sessions_dir("default") / "old1.json"
    assert dest.is_file()
    assert json.loads(dest.read_text(encoding="utf-8"))["title"] == "旧会话"
    assert (tmp_path / ".sessions.migrated").is_dir()
    assert not legacy.exists()
    copied = svc.migrate_legacy_sessions(legacy)
    assert copied == 0
    assert dest.is_file()
    svc2 = AgentService(tmp_path / "agents", legacy_sessions=legacy)
    assert (svc2.sessions_dir("default") / "old1.json").is_file()


def test_session_persists_under_agent_and_reloads(
    tmp_path: Path,
    mock_llm_factory: type,
) -> None:
    svc = _svc(tmp_path)
    cfg = AgentConfig(max_steps=2, planner_enabled=False, verifier_enabled=False)
    llm = mock_llm_factory(["第一句回答"])
    runner = AgentRunner(cfg, llm, ToolRegistry(cfg))
    mgr = SessionManager(cfg, llm, runner, sessions_dir=None, agents=svc)
    mgr.chat("s1", "你好", agent="default")
    path = svc.sessions_dir("default") / "s1.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["agent"] == "default"
    assert data["turns"][0]["answer"] == "第一句回答"

    llm2 = mock_llm_factory([-1, "第二句"])
    mgr2 = SessionManager(
        cfg,
        llm2,
        AgentRunner(cfg, llm2, ToolRegistry(cfg)),
        agents=svc,
    )
    state = mgr2.get_state("s1", agent="default")
    assert len(state.turns) == 1
    assert state.turns[0].answer == "第一句回答"


def test_sessions_isolated_per_agent(
    tmp_path: Path,
    mock_llm_factory: type,
) -> None:
    svc = _svc(tmp_path)
    svc.create("写手")
    cfg = AgentConfig(max_steps=2, planner_enabled=False, verifier_enabled=False)
    llm = mock_llm_factory(["默认答", "写手答"])
    mgr = SessionManager(
        cfg, llm, AgentRunner(cfg, llm, ToolRegistry(cfg)), agents=svc
    )
    mgr.chat("s1", "你好", agent="default")
    mgr.chat("s1", "开写", agent="写手")
    assert mgr.get_state("s1", agent="default").turns[0].answer == "默认答"
    assert mgr.get_state("s1", agent="写手").turns[0].answer == "写手答"
    assert (svc.sessions_dir("default") / "s1.json").is_file()
    assert (svc.sessions_dir("写手") / "s1.json").is_file()
    assert [row["agent"] for row in mgr.list_sessions("写手")] == ["写手"]


def test_entities_never_compressed_into_summary(
    tmp_path: Path,
    mock_llm_factory: type,
) -> None:
    svc = _svc(tmp_path)
    svc.write_entities("default", {"工号": "X-99", "姓名": "林小满"})
    cfg = AgentConfig(
        max_steps=2,
        planner_enabled=False,
        verifier_enabled=False,
        keep_full_turns=1,
        session_summary_limit=40,
    )
    llm = mock_llm_factory(["答一", "改写后的第二轮", "答二", "摘要应丢掉实体"])
    mgr = SessionManager(
        cfg, llm, AgentRunner(cfg, llm, ToolRegistry(cfg)), agents=svc
    )
    mgr.chat("s1", "第一轮闲聊很多字好让摘要启动", agent="default")
    mgr.chat("s1", "第二轮还要继续聊", agent="default")
    assert "X-99" in mgr.last_run_task
    assert "林小满" in mgr.last_run_task
    assert "X-99" not in (mgr.get_state("s1", agent="default").rolling_summary)


def test_prompt_md_injected_into_planner(
    tmp_path: Path,
    mock_llm_factory: type,
) -> None:
    svc = _svc(tmp_path)
    svc.write_prompt("default", "你是写手人格，只写小说。")

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            del event, payload

    cfg = AgentConfig(max_steps=2, verifier_enabled=False)
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "闲聊",
                "constraints": {},
                "plan": ["答"],
                "needs_history": False,
            },
            "你好呀",
            "标题",
        ]
    )
    mgr = SessionManager(
        cfg,
        llm,
        AgentRunner(cfg, llm, ToolRegistry(cfg)),
        events=_Bus(),
        agents=svc,
    )
    mgr.chat("s1", "你好", agent="default")
    assert any("写手人格" in p for p in llm.prompts)


def test_skill_hit_injected(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    skill_dir = svc.agent_dir("default") / "skills" / "outline"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# 大纲技能\nkeywords: 大纲\n写长文前先列大纲。\n",
        encoding="utf-8",
    )
    hits = svc.match_skills("default", "请按大纲技能写一篇")
    assert hits and "先列大纲" in hits[0].body
    preamble = svc.system_preamble("default", "请按大纲技能写一篇")
    assert "先列大纲" in preamble


def test_session_plugin_warns_without_persist(caplog: pytest.LogCaptureFixture) -> None:
    ctx = Context()

    class _Llm:
        def generate(self, p: str) -> str:
            return "x"

        def generate_int(self, p: str, a: int, b: int) -> int:
            return -1

        def generate_json(self, p: str, s: dict) -> dict:
            return {}

    class _LlmP:
        name = "llm"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["llm"]

        def apply(self, c: Context, config: dict) -> None:
            c.provide("llm", _Llm())

    class _AgentP:
        name = "agent-loop"
        requires: ClassVar[list[str]] = ["llm"]
        provides: ClassVar[list[str]] = ["agent"]

        def apply(self, c: Context, config: dict) -> None:
            class _Runner:
                pass

            c.provide("agent", _Runner())

    ctx.mount(_LlmP())
    ctx.mount(_AgentP())
    with caplog.at_level(logging.WARNING):
        ctx.mount(SessionPlugin())
    assert any("会话未持久化" in rec.message for rec in caplog.records)
    assert ctx.require("sessions").persist_enabled is False


def test_agents_plugin_creates_tree(tmp_path: Path) -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    ctx.mount(
        AgentsPlugin(),
        {
            "agents_root": str(tmp_path / "agents"),
            "legacy_sessions": str(tmp_path / "no_legacy"),
        },
    )
    svc = ctx.require("agents")
    assert (svc.root() / "shared" / "memory").is_dir()
    assert (svc.root() / "default" / "agent.yml").is_file()
    ctx.unmount("agents")
    assert not ctx.has("agents")
