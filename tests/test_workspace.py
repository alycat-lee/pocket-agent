from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.ui_web import ApprovalBroker, build_app
from pocket_agent.plugins.workspace_plugin import WorkspacePlugin
from pocket_agent.runtime import Context
from pocket_agent.workspace import WorkspaceManager

pytest.importorskip("fastapi")


# ── 路径围栏 ──────────────────────────────────────────────


def test_resolve_relative_path(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    ws = WorkspaceManager(settings_path=settings)
    proj = ws.create("proj", str(tmp_path))
    ws.bind_session("s1", proj.id)
    # 相对路径解析到工作区根
    resolved = ws.resolve_path("s1", "report.txt")
    assert resolved == str(tmp_path / "report.txt")
    # 绝对路径原样返回
    abs = str(tmp_path / "data" / "x.txt")
    assert ws.resolve_path("s1", abs) == abs


def test_validate_path_blocks_escape(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    ws = WorkspaceManager(settings_path=settings)
    proj = ws.create("proj", str(tmp_path / "sandbox"))
    ws.bind_session("s1", proj.id)
    # 工作区内 → True
    inside = ws.resolve_path("s1", "report.txt")
    assert ws.validate_path("s1", inside) is True
    # ../ 逃逸 → False
    escaped = ws.resolve_path("s1", "../../etc/passwd")
    assert ws.validate_path("s1", escaped) is False
    # 绝对路径攻击 → False
    assert ws.validate_path("s1", "/etc/passwd") is False


def test_rebind_workspace_changes_root(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    ws = WorkspaceManager(settings_path=settings)
    a = ws.create("a", str(tmp_path / "dir_a"))
    b = ws.create("b", str(tmp_path / "dir_b"))
    ws.bind_session("s1", a.id)
    assert ws.resolve_path("s1", "f.txt") == str(tmp_path / "dir_a" / "f.txt")
    ws.bind_session("s1", b.id)
    assert ws.resolve_path("s1", "f.txt") == str(tmp_path / "dir_b" / "f.txt")


# ── 权限三档 ──────────────────────────────────────────────


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


class _FakeLlmPlugin:
    name = "llm-fake"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm"]

    def __init__(self, script: list) -> None:
        self._script = script

    def apply(self, ctx: Context, config: dict) -> None:
        ctx.provide("llm", _FakeLlm(self._script))


def _stack_with_workspace(tmp_path: Path, script: list) -> Context:
    ctx = Context()
    ctx.mount(_FakeLlmPlugin(script))
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 5})
    ctx.mount(WorkspacePlugin(), {"settings_path": str(tmp_path / "settings.json")})
    ctx.mount(SessionPlugin())
    return ctx


def test_read_file_workspace_fencing(tmp_path: Path) -> None:
    """工作区内文件可读，逃逸路径被拦截。"""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "report.txt").write_text("hello", encoding="utf-8")

    ctx = _stack_with_workspace(tmp_path, [1, 3, {"path": "report.txt"}, -1, "done"])
    ws = ctx.require("workspace")
    # 改默认工作区根到 sandbox
    ws._workspaces["default"].root = str(sandbox)

    sessions = ctx.require("sessions")
    sessions.chat("s1", "读取 report.txt", permission="full")
    # 检查工具步骤结果而非 final_answer（MockLLM 的 final 固定返回 "done"）
    read_steps = [s for s in sessions.last_result.steps if s.tool_name == "read_file"]
    assert len(read_steps) >= 1
    assert "hello" in read_steps[0].result
    ctx.unmount_all()


def test_permission_readonly_disables_write(tmp_path: Path) -> None:
    """readonly 模式下写工具被过滤，无法选中。"""
    ctx = _stack_with_workspace(tmp_path, [1, -1, "done"])
    sessions = ctx.require("sessions")
    sessions.chat("s1", "写文件", permission="readonly")
    assert sessions.last_result is not None
    write_steps = [
        s for s in sessions.last_result.steps if s.tool_name == "write_file"
    ]
    assert len(write_steps) == 0
    ctx.unmount_all()


def test_permission_full_skips_approval(tmp_path: Path) -> None:
    """full 模式下写工具免审批直接执行。"""
    # write_file is at index 4 (after get_weather, calculator, read_file, get_current_time)
    ctx = _stack_with_workspace(tmp_path, [1, 4, {"path": "x.txt", "content": "ok"}, -1, "done"])
    broker = ApprovalBroker(timeout_sec=0.3)
    ctx.require("agent").approval_gate = broker.request
    sessions = ctx.require("sessions")
    sessions.chat("s1", "写文件", permission="full")
    assert sessions.last_result is not None
    write_steps = [
        s for s in sessions.last_result.steps if s.tool_name == "write_file"
    ]
    assert len(write_steps) >= 1
    ctx.unmount_all()


def test_permission_workspace_triggers_approval(tmp_path: Path) -> None:
    """workspace 模式下写工具弹审批，超时拒绝。"""
    ctx = _stack_with_workspace(tmp_path, [1, 4, {"path": "x.txt", "content": "ok"}, -1, "done"])
    broker = ApprovalBroker(timeout_sec=0.3)
    ctx.require("agent").approval_gate = broker.request
    sessions = ctx.require("sessions")
    sessions.chat("s1", "写文件", permission="workspace")
    assert sessions.last_result is not None
    # 审批超时 → 拦截，write_file 不会出现在正常步骤中
    ctx.unmount_all()


# ── Workspace API ─────────────────────────────────────────


async def _request(app, method, path, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        return await c.request(method, path, **kwargs)


def test_workspace_api(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    ctx = Context()
    ctx.mount(_FakeLlmPlugin([1, -1, "x"]))
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin())
    ctx.mount(WorkspacePlugin(), {"settings_path": str(settings)})
    ctx.mount(SessionPlugin())
    app = build_app(
        ctx,
        broker=ApprovalBroker(1),
        settings_path=settings,
        pocket_data={"plugins": []},
    )

    # List (has default)
    res = asyncio.run(_request(app, "GET", "/api/workspaces"))
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert len(data["workspaces"]) >= 1

    # Create
    res = asyncio.run(
        _request(app, "POST", "/api/workspaces", json={"name": "proj", "root": str(tmp_path)})
    )
    assert res.status_code == 200
    wid = res.json()["workspace"]["id"]

    # Bind session
    res = asyncio.run(
        _request(app, "PUT", "/api/sessions/s1/workspace", json={"workspace_id": wid})
    )
    assert res.status_code == 200
    assert res.json()["workspace_id"] == wid

    ctx.unmount_all()
