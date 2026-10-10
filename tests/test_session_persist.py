from __future__ import annotations

import json
from pathlib import Path

from pocket_agent.config import AgentConfig
from pocket_agent.runner import AgentRunner
from pocket_agent.session import SessionManager
from pocket_agent.tools import ToolRegistry


class _FakeLlm:
    def __init__(self, script: list) -> None:
        self._script = list(script)

    def generate(self, p: str) -> str:
        return str(self._script.pop(0) if self._script else "")

    def generate_int(self, p: str, minimum: int, maximum: int) -> int:
        return int(self._script.pop(0) if self._script else -1)

    def generate_json(self, p: str, s: dict) -> dict:
        v = self._script.pop(0) if self._script else {}
        assert isinstance(v, dict)
        return v


def _mgr(tmp_path: Path, script: list) -> SessionManager:
    cfg = AgentConfig(max_steps=3)
    registry = ToolRegistry(cfg)
    llm = _FakeLlm(script)
    runner = AgentRunner(cfg, llm, registry)
    return SessionManager(
        cfg, llm, runner, sessions_dir=tmp_path / ".sessions"
    )


def test_session_persist_and_reload(tmp_path: Path) -> None:
    """写盘后重建 SessionManager 能恢复 title 与 turns。"""
    mgr = _mgr(tmp_path, ["回答1"])
    mgr.chat("s1", "你好")
    assert mgr.get_state("s1").title == "新会话"
    assert len(mgr.get_state("s1").turns) == 1

    # 重建：从磁盘加载
    mgr2 = _mgr(tmp_path, ["回答2"])
    state = mgr2.get_state("s1")
    assert state.title == "新会话"
    assert len(state.turns) == 1
    assert state.turns[0].answer == "回答1"


def test_corrupt_session_skipped(tmp_path: Path) -> None:
    """损坏的 session 文件跳过不崩溃。"""
    sessions_dir = tmp_path / ".sessions"
    sessions_dir.mkdir()
    (sessions_dir / "bad.json").write_text("{invalid json", encoding="utf-8")
    (sessions_dir / "good.json").write_text(
        json.dumps({"session_id": "good", "title": "好的", "turns": []}),
        encoding="utf-8",
    )

    mgr = _mgr(tmp_path, [-1, "x"])
    assert "good" in mgr._sessions
    assert "bad" not in mgr._sessions
    assert mgr.get_state("good").title == "好的"


def test_list_sessions_sorted_by_updated(tmp_path: Path) -> None:
    """list_sessions 按 updated_at 降序返回。"""
    mgr = _mgr(tmp_path, ["a"])
    mgr.chat("old", "first")
    mgr.get_state("old").updated_at = 1000
    mgr._save_session(mgr.get_state("old"))
    mgr.chat("new", "second")

    rows = mgr.list_sessions()
    assert rows[0]["session_id"] == "new"
    assert rows[1]["session_id"] == "old"


def test_delete_session(tmp_path: Path) -> None:
    """删除会话后文件消失。"""
    mgr = _mgr(tmp_path, ["x"])
    mgr.chat("s1", "test")
    assert (tmp_path / ".sessions" / "s1.json").is_file()
    mgr.delete_session("s1")
    assert not (tmp_path / ".sessions" / "s1.json").is_file()
    assert "s1" not in mgr._sessions
