from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from pocket_agent.config import AgentConfig
from pocket_agent.ledger import CallLedger
from pocket_agent.plugins import builtin_plugins
from pocket_agent.plugins.workspace_plugin import (
    fenced_edit_file,
    fenced_grep_files,
    fenced_list_dir,
    fenced_read_file,
    fenced_write_file,
)
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context, assemble
from pocket_agent.tool_context import set_session_id
from pocket_agent.tools import ToolRegistry, tool
from pocket_agent.workspace import WorkspaceManager


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))


class _FakeLlm:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)

    def generate(self, prompt: str) -> str:
        return str(self._script.pop(0) if self._script else "")

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return int(self._script.pop(0) if self._script else -1)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        item = self._script.pop(0) if self._script else {}
        assert isinstance(item, dict)
        return item


class _FakeLlmAsLlama:
    name = "llm-llamacpp"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm", "llm_local"]

    def apply(self, ctx: Context, config: dict) -> None:
        llm = _FakeLlm([-1, "ok"])
        ctx.provide("llm_local", llm)
        ctx.provide("llm", llm)


def _manager(tmp_path: Path) -> tuple[WorkspaceManager, Path]:
    settings = tmp_path / "settings.json"
    manager = WorkspaceManager(settings_path=settings)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    manager._workspaces["default"].root = str(sandbox)
    return manager, sandbox


def test_write_file_atomic_and_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, sandbox = _manager(tmp_path)
    writes: list[str] = []
    replaces: list[tuple[str, str]] = []
    orig_write = Path.write_text
    orig_replace = Path.replace

    def spy_write(self: Path, data: str, *args: Any, **kwargs: Any) -> None:
        writes.append(str(self))
        orig_write(self, data, *args, **kwargs)

    def spy_replace(self: Path, target: Path) -> Path:
        replaces.append((str(self), str(target)))
        return orig_replace(self, target)

    monkeypatch.setattr(Path, "write_text", spy_write)
    monkeypatch.setattr(Path, "replace", spy_replace)

    msg = fenced_write_file(manager, "s1", "hello.md", "你好")
    assert msg == "已写入 2 字符：hello.md"
    assert (sandbox / "hello.md").read_text(encoding="utf-8") == "你好"
    assert any(item.endswith("hello.md.tmp") for item in writes)
    assert any(dest.endswith("hello.md") for _src, dest in replaces)

    escaped = fenced_write_file(manager, "s1", "../outside.txt", "x")
    assert escaped == (
        f"[工具错误] 路径 ../outside.txt 超出工作区 {sandbox}，请使用工作区内相对路径"
    )
    assert not (tmp_path / "outside.txt").exists()


def test_edit_file_three_branches_and_escape(tmp_path: Path) -> None:
    manager, sandbox = _manager(tmp_path)
    target = sandbox / "note.md"
    target.write_text("你好世界你好", encoding="utf-8")

    missing = fenced_edit_file(manager, "s1", "nope.md", "a", "b")
    assert missing == "文件不存在：nope.md"

    zero = fenced_edit_file(manager, "s1", "note.md", "不存在", "x")
    assert zero == "[工具错误] 未找到匹配文本，请先用 read_file 确认原文"

    many = fenced_edit_file(manager, "s1", "note.md", "你好", "嗨")
    assert many == "[工具错误] 匹配到 2 处，请在 old 中补充更多上下文使其唯一"
    assert target.read_text(encoding="utf-8") == "你好世界你好"

    ok = fenced_edit_file(manager, "s1", "note.md", "你好世界你好", "你好世界")
    assert ok == "已替换：note.md（6 → 4 字符）"
    assert target.read_text(encoding="utf-8") == "你好世界"

    escaped = fenced_edit_file(manager, "s1", "/etc/passwd", "a", "b")
    assert escaped == (
        f"[工具错误] 路径 /etc/passwd 超出工作区 {sandbox}，请使用工作区内相对路径"
    )


def test_edit_file_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, sandbox = _manager(tmp_path)
    (sandbox / "a.txt").write_text("only-once", encoding="utf-8")
    writes: list[str] = []
    orig_write = Path.write_text

    def spy_write(self: Path, data: str, *args: Any, **kwargs: Any) -> None:
        writes.append(str(self))
        orig_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write)
    fenced_edit_file(manager, "s1", "a.txt", "only-once", "done")
    assert any(item.endswith("a.txt.tmp") for item in writes)


def test_list_dir_nested_skip_and_truncate(tmp_path: Path) -> None:
    manager, sandbox = _manager(tmp_path)
    (sandbox / "root.txt").write_text("x", encoding="utf-8")
    nested = sandbox / "sub" / "deep"
    nested.mkdir(parents=True)
    (sandbox / "sub" / "a.txt").write_text("aa", encoding="utf-8")
    (nested / "too-deep.txt").write_text("z", encoding="utf-8")
    (sandbox / ".git").mkdir()
    (sandbox / ".git" / "config").write_text("x", encoding="utf-8")
    (sandbox / "node_modules").mkdir()
    (sandbox / "__pycache__").mkdir()

    listed = fenced_list_dir(manager, "s1", ".")
    assert "[file] root.txt 1B" in listed
    assert "[dir] sub/" in listed
    assert "[file] sub/a.txt 2B" in listed
    assert "too-deep" not in listed
    assert ".git" not in listed
    assert "node_modules" not in listed
    assert "__pycache__" not in listed

    crowded = sandbox / "many"
    crowded.mkdir()
    for i in range(210):
        (crowded / f"f{i:03d}.txt").write_text("1", encoding="utf-8")
    truncated = fenced_list_dir(manager, "s1", "many")
    assert "…（共 210 条，已截断）" in truncated
    assert truncated.count("[file]") == 200


def test_grep_files_hit_miss_invalid_binary(tmp_path: Path) -> None:
    manager, sandbox = _manager(tmp_path)
    (sandbox / "hit.txt").write_text("alpha\nbeta\nalpha2\n", encoding="utf-8")
    (sandbox / "skip.bin").write_bytes(b"\x00\x01\x02alpha")
    (sandbox / ".git").mkdir()
    (sandbox / ".git" / "secret.txt").write_text("alpha hidden", encoding="utf-8")

    hit = fenced_grep_files(manager, "s1", "alpha", ".")
    assert "hit.txt:1: alpha" in hit
    assert "hit.txt:3: alpha2" in hit
    assert "skip.bin" not in hit
    assert "secret.txt" not in hit

    miss = fenced_grep_files(manager, "s1", "zzz", ".")
    assert miss == "（无匹配）"

    bad = fenced_grep_files(manager, "s1", "(", ".")
    assert bad.startswith("[工具错误] 正则无效：")


def test_headless_assemble_registers_fenced_workspace(tmp_path: Path) -> None:
    yml = {
        "profile": "headless",
        "plugins": [
            {"name": "llm-llamacpp", "config": {"base_url": "http://127.0.0.1:9/v1"}},
            {"name": "tools-core"},
            {"name": "workspace", "config": {"settings_path": str(tmp_path / "s.json")}},
            {
                "name": "knowledge-base",
                "config": {
                    "embedder": "hash",
                    "db_path": str(tmp_path / "kb.sqlite"),
                },
            },
            {"name": "agent-loop", "config": {"max_steps": 2}},
            {"name": "session"},
            {"name": "ui-web", "config": {"host": "127.0.0.1", "port": 3997}},
        ],
    }
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, yml)
    try:
        tools = ctx.require("tools")
        names = {item.name for item in tools.all()}
        assert "write_file" in names
        assert "edit_file" in names
        assert "list_dir" in names
        assert "grep_files" in names
        assert tools.get("read_file").short_desc == "读取工作区内文件"
        set_session_id("assemble")
        denied = tools.get("read_file").fn(path="/etc/passwd")
        assert "超出工作区" in denied
        assert "请使用工作区内相对路径" in denied
        assert "workspace" in ctx.mount_order
        assert "ui-web" not in ctx.mount_order
    finally:
        ctx.unmount_all()


def _write_registry(tmp_path: Path) -> tuple[ToolRegistry, Path]:
    manager, sandbox = _manager(tmp_path)
    registry = ToolRegistry(AgentConfig(max_steps=5, tool_recall_threshold=20))

    def write_file(path: str, content: str) -> str:
        return fenced_write_file(manager, "s1", path, content)

    def edit_file(path: str, old: str, new: str) -> str:
        return fenced_edit_file(manager, "s1", path, old, new)

    def read_file(path: str) -> str:
        return fenced_read_file(manager, "s1", path)

    registry.register(read_file, "读取工作区内文件", cacheable=True)
    registry.register(write_file, "写入工作区内文件", cacheable=False)
    registry.register(edit_file, "替换文件中的文本", cacheable=False)
    return registry, sandbox


def test_fenced_read_file_missing_sets_error_and_hint(tmp_path: Path) -> None:
    registry, _sandbox = _write_registry(tmp_path)
    llm = _FakeLlm(
        [
            0,
            {"path": "a.md"},
            0,
            {"path": "b.md"},
            -1,
            "done",
        ]
    )
    result = AgentRunner(
        AgentConfig(
            max_steps=6,
            tool_recall_threshold=20,
            max_consecutive_intercepts=10,
        ),
        llm,
        registry,
    ).run("读取文件 a.md 与 b.md", session_id="s1")
    reads = [s for s in result.steps if s.tool_name == "read_file"]
    assert len(reads) == 2
    assert reads[0].error is not None and "文件不存在" in reads[0].error
    assert reads[1].error is not None and "文件不存在" in reads[1].error
    assert "该路径不存在" in reads[1].result
    assert "其他工具" in reads[1].result


def test_runner_write_denied_does_not_create_file(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    bus = _Bus()
    llm = _FakeLlm([1, {"path": "x.txt", "content": "secret"}, -1, "done"])
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        llm,
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: False
    result = runner.run("写文件", permission="workspace")
    assert not (sandbox / "x.txt").exists()
    assert any(name == "approval.required" for name, _p in bus.events)
    assert any(
        payload.get("reason") == "approval_denied" for _n, payload in bus.events
    )
    assert any(step.error and "拒绝" in step.error for step in result.steps)


def test_runner_write_approved_and_edit(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    bus = _Bus()
    llm = _FakeLlm([1, {"path": "x.txt", "content": "你好"}, -1, "done"])
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        llm,
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: True
    runner.run("写文件", permission="workspace")
    assert (sandbox / "x.txt").read_text(encoding="utf-8") == "你好"
    assert any(name == "approval.required" for name, _p in bus.events)

    llm2 = _FakeLlm([2, {"path": "x.txt", "old": "你好", "new": "你好世界"}, -1, "done"])
    runner2 = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        llm2,
        registry,
        events=_Bus(),
    )
    runner2.approval_gate = lambda _payload: True
    runner2.run("改文件", permission="workspace")
    assert (sandbox / "x.txt").read_text(encoding="utf-8") == "你好世界"


def test_runner_full_skips_approval(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    bus = _Bus()
    llm = _FakeLlm([1, {"path": "y.txt", "content": "full"}, -1, "done"])
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        llm,
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: False
    runner.run("写文件", permission="full")
    assert (sandbox / "y.txt").read_text(encoding="utf-8") == "full"
    assert all(name != "approval.required" for name, _p in bus.events)


def test_runner_ledger_blocks_repeat_write(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    ledger = CallLedger()
    config = AgentConfig(max_steps=4, tool_recall_threshold=20)
    args = {"path": "z.txt", "content": "once"}
    first = AgentRunner(config, _FakeLlm([1, args, -1, "done"]), registry)
    first.run("写文件", ledger=ledger, permission="full")
    assert (sandbox / "z.txt").read_text(encoding="utf-8") == "once"
    assert ledger.real_call_count == 1
    (sandbox / "z.txt").write_text("tampered", encoding="utf-8")
    second = AgentRunner(config, _FakeLlm([1, args, -1, "done"]), registry)
    result = second.run("再写同一文件", ledger=ledger, permission="full")
    assert ledger.real_call_count == 1
    assert result.stats["cache_hits"] >= 1
    assert (sandbox / "z.txt").read_text(encoding="utf-8") == "tampered"


def test_runner_readonly_excludes_write_tools(tmp_path: Path) -> None:
    manager, sandbox = _manager(tmp_path)
    registry = ToolRegistry(AgentConfig(max_steps=4, tool_recall_threshold=20))

    def write_file(path: str, content: str) -> str:
        return fenced_write_file(manager, "s1", path, content)

    def read_file(path: str) -> str:
        return fenced_read_file(manager, "s1", path)

    registry.register(write_file, "写入工作区内文件", cacheable=False)
    registry.register(read_file, "读取工作区内文件", cacheable=True)
    llm = _FakeLlm([0, {"path": "nope.txt", "content": "x"}, -1, "done"])
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        llm,
        registry,
    )
    result = runner.run("写文件", permission="readonly")
    assert all(step.tool_name != "write_file" for step in result.steps)
    assert not (sandbox / "nope.txt").exists()


def test_recall_write_file_when_tools_grow_to_ten() -> None:
    config = AgentConfig(tool_recall_threshold=8, tool_recall_top_n=5)
    registry = ToolRegistry(config)

    @tool(registry, "查询天气")
    def get_weather(city: str) -> str:
        return city

    @tool(registry, "做计算")
    def calculator(expression: str) -> str:
        return expression

    @tool(registry, "取时间")
    def get_current_time() -> str:
        return "t"

    @tool(registry, "读知识")
    def search_knowledge(query: str) -> str:
        return query

    @tool(registry, "查百科")
    def wiki_lookup(query: str) -> str:
        return query

    @tool(registry, "搜网页")
    def web_search(query: str) -> str:
        return query

    assert len(registry.all()) == 6
    assert len(registry.recall("保存文件", top_n=5)) == 6

    @tool(registry, "写入工作区内文件", cacheable=False)
    def write_file(path: str, content: str) -> str:
        return path

    @tool(registry, "替换文件中的文本", cacheable=False)
    def edit_file(path: str, old: str, new: str) -> str:
        return path

    @tool(registry, "列出工作区目录")
    def list_dir(path: str = ".") -> str:
        return path

    @tool(registry, "搜索工作区文本")
    def grep_files(pattern: str, path: str = ".") -> str:
        return pattern

    assert len(registry.all()) == 10
    recalled = registry.recall("保存文件", top_n=5)
    assert len(recalled) == 5
    assert any(item.name == "write_file" for item in recalled)
