from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, ClassVar

from pocket_agent.config import AgentConfig
from pocket_agent.plugins import builtin_plugins
from pocket_agent.plugins.shell_plugin import (
    ShellPlugin,
    run_command_in,
)
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.workspace_plugin import WorkspacePlugin
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context, assemble
from pocket_agent.tool_context import set_session_id
from pocket_agent.tools import ToolRegistry
from pocket_agent.workspace import WorkspaceManager


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


def test_run_command_echo_hello(tmp_path: Path) -> None:
    text = run_command_in(
        "echo hello",
        cwd=str(tmp_path),
        timeout_sec=30,
        max_output_chars=4000,
    )
    assert text.startswith("退出码 0\n")
    assert "hello" in text


def test_run_command_nonzero_exit(tmp_path: Path) -> None:
    cmd = f'"{sys.executable}" -c "raise SystemExit(3)"'
    text = run_command_in(cmd, cwd=str(tmp_path), timeout_sec=30, max_output_chars=4000)
    assert text.startswith("退出码 3\n")


def test_run_command_timeout(tmp_path: Path) -> None:
    command = "ping -n 20 127.0.0.1" if os.name == "nt" else "sleep 20"
    text = run_command_in(
        command,
        cwd=str(tmp_path),
        timeout_sec=1,
        max_output_chars=4000,
    )
    assert text == "[工具错误] 命令超时（1s），进程已终止"


def test_run_command_truncates_output(tmp_path: Path) -> None:
    cmd = f'"{sys.executable}" -c "print(\'x\' * 5000)"'
    text = run_command_in(cmd, cwd=str(tmp_path), timeout_sec=30, max_output_chars=100)
    assert text.startswith("退出码 0\n")
    assert "…（输出过长已截断，共 " in text
    assert "字符）" in text
    body = text.split("\n", 1)[1]
    assert body.startswith("x" * 100)
    assert "共 " in text and "字符）" in text


def test_run_command_cwd_is_workspace(tmp_path: Path) -> None:
    cmd = "cd" if os.name == "nt" else "pwd"
    text = run_command_in(cmd, cwd=str(tmp_path), timeout_sec=30, max_output_chars=4000)
    assert text.startswith("退出码 0\n")
    expected = str(tmp_path.resolve())
    normalized = text.replace("/", os.sep).replace("\\", os.sep)
    assert expected in normalized or expected.casefold() in normalized.casefold()


def test_readonly_excludes_run_command(tmp_path: Path) -> None:
    registry = ToolRegistry(AgentConfig(max_steps=4, tool_recall_threshold=20))

    def run_command(command: str) -> str:
        (tmp_path / "ran.txt").write_text(command, encoding="utf-8")
        return command

    def get_current_time() -> str:
        return "now"

    registry.register(run_command, "在工作区执行命令", cacheable=False)
    registry.register(get_current_time, "获取当前时间", cacheable=True)
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        _FakeLlm([0, {}, -1, "done"]),
        registry,
    )
    result = runner.run("运行命令", permission="readonly")
    assert all(step.tool_name != "run_command" for step in result.steps)
    assert not (tmp_path / "ran.txt").exists()


def test_shell_plugin_unmount_removes_tool(tmp_path: Path) -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    ctx.mount(WorkspacePlugin(), {"settings_path": str(tmp_path / "s.json")})
    ctx.mount(ShellPlugin(), {"timeout_sec": 15, "max_output_chars": 200})
    tools = ctx.require("tools")
    assert "run_command" in {item.name for item in tools.all()}
    tool = tools.get("run_command")
    assert tool.cacheable is False
    assert len(tool.short_desc) <= 20
    ctx.unmount("shell")
    assert "run_command" not in {item.name for item in tools.all()}


def test_headless_assemble_registers_run_command(tmp_path: Path) -> None:
    yml = {
        "profile": "headless",
        "plugins": [
            {"name": "llm-llamacpp", "config": {"base_url": "http://127.0.0.1:9/v1"}},
            {"name": "tools-core"},
            {"name": "workspace", "config": {"settings_path": str(tmp_path / "s.json")}},
            {"name": "shell"},
            {"name": "agent-loop", "config": {"max_steps": 2}},
            {"name": "session"},
            {"name": "ui-web", "config": {"host": "127.0.0.1", "port": 3996}},
        ],
    }
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, yml)
    try:
        tools = ctx.require("tools")
        names = {item.name for item in tools.all()}
        assert "run_command" in names
        assert tools.get("run_command").cacheable is False
        assert "shell" in ctx.mount_order
        manager: WorkspaceManager = ctx.require("workspace")
        sandbox = tmp_path / "ws"
        sandbox.mkdir()
        manager._workspaces["default"].root = str(sandbox)
        set_session_id("assemble")
        text = tools.get("run_command").fn(command="echo hello")
        assert "hello" in text
        assert text.startswith("退出码 0\n")
    finally:
        ctx.unmount_all()


def test_builtin_plugins_includes_shell() -> None:
    assert "shell" in builtin_plugins()
