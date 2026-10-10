from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import pytest
from pocket_agent.plugins.bridge_mcp import (
    BridgeMcpPlugin,
    StdioMcpClient,
    adapt_mcp_tool,
    content_to_str,
    is_write_tool,
    mcp_tool_name,
    truncate_result,
)
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.runtime import Context
from pocket_agent.tools import ToolRegistry

_REPO = Path(__file__).resolve().parents[1]
_MOCK = _REPO / "scripts" / "mock_mcp_echo.py"


def _stdio_spec(**extra: object) -> dict:
    return {
        "name": "echo",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(_MOCK)],
        **extra,
    }


def test_write_tool_heuristic() -> None:
    assert is_write_tool("write_note")
    assert is_write_tool("fs_delete_file")
    assert is_write_tool("createUser")
    assert not is_write_tool("echo")
    assert not is_write_tool("get_weather")
    assert not is_write_tool("list_dir")


def test_mcp_tool_name_prefix() -> None:
    assert mcp_tool_name("echo") == "mcp_echo"
    assert mcp_tool_name("mcp_echo") == "mcp_echo"
    assert mcp_tool_name("a/b c") == "mcp_a_b_c"


def test_content_to_str_and_truncate() -> None:
    text = content_to_str(
        {"content": [{"type": "text", "text": "hello"}], "isError": False}
    )
    assert text == "hello"
    assert truncate_result("abcdef", 3) == "abc"
    err = content_to_str(
        {"content": [{"type": "text", "text": "boom"}], "isError": True}
    )
    assert err.startswith("[MCP错误]")


def test_stdio_list_and_call_roundtrip() -> None:
    assert _MOCK.is_file()
    client = StdioMcpClient(
        sys.executable,
        [str(_MOCK)],
        name="echo",
        timeout=10,
    )
    try:
        assert client.alive()
        assert client.pid is not None
        tools = client.list_tools()
        names = {t["name"] for t in tools}
        assert "echo" in names
        assert "write_note" in names
        result = client.call_tool("echo", {"text": "ping"})
        assert content_to_str(result) == "echo:ping"
    finally:
        client.close()
    deadline = time.time() + 3
    while client.alive() and time.time() < deadline:
        time.sleep(0.05)
    assert not client.alive()


def test_adapt_registers_prefix_and_cacheable() -> None:
    client = StdioMcpClient(sys.executable, [str(_MOCK)], name="echo", timeout=10)
    try:
        tools = client.list_tools()
        by_name = {t["name"]: t for t in tools}
        echo = adapt_mcp_tool(client, by_name["echo"], per_result_limit=300)
        write = adapt_mcp_tool(client, by_name["write_note"], per_result_limit=300)
        assert echo.name == "mcp_echo"
        assert echo.cacheable is True
        assert write.name == "mcp_write_note"
        assert write.cacheable is False
        assert echo.fn(text="hi") == "echo:hi"
        assert write.fn(text="x") == "wrote:x"
    finally:
        client.close()


def test_source_compress_limit_on_call() -> None:
    client = StdioMcpClient(sys.executable, [str(_MOCK)], name="echo", timeout=10)
    try:
        tool = adapt_mcp_tool(
            client,
            {
                "name": "echo",
                "description": "回显",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            per_result_limit=9,
        )
        out = tool.fn(text="123456789012")
        assert out == "echo:1234"
        assert len(out) == 9
    finally:
        client.close()


def test_bridge_plugin_registers_tools() -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    ctx.mount(
        BridgeMcpPlugin(),
        {"per_result_limit": 50, "mcp_servers": [_stdio_spec()]},
    )
    registry: ToolRegistry = ctx.require("tools")
    names = {t.name for t in registry.all()}
    assert "mcp_echo" in names
    assert "mcp_write_note" in names
    assert "get_weather" in names
    assert registry.get("mcp_echo").fn(text="z") == "echo:z"
    assert registry.get("mcp_write_note").cacheable is False
    ctx.unmount_all()


def _pid_running(pid: int) -> bool:
    """跨平台：进程是否仍在运行（Windows 用 GetExitCodeProcess）。"""
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        if not ok:
            return False
        return int(code.value) == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_unmount_removes_tools_and_kills_subprocess() -> None:
    """卸载 bridge-mcp 后工具消失，且 stdio 子进程真正退出。"""
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    plugin = BridgeMcpPlugin()
    ctx.mount(plugin, {"mcp_servers": [_stdio_spec(name="killme")]})
    registry: ToolRegistry = ctx.require("tools")
    assert "mcp_echo" in {t.name for t in registry.all()}
    assert registry.get("mcp_echo").fn(text="ok") == "echo:ok"

    assert len(plugin.clients) == 1
    child = plugin.clients[0]
    assert isinstance(child, StdioMcpClient)
    pid = child.pid
    assert pid is not None
    assert child.alive()
    assert _pid_running(pid)

    ctx.unmount("bridge-mcp")

    names_after = {t.name for t in registry.all()}
    assert "mcp_echo" not in names_after
    assert "mcp_write_note" not in names_after
    assert "get_weather" in names_after
    with pytest.raises(KeyError):
        registry.get("mcp_echo")

    deadline = time.time() + 5
    while (child.alive() or _pid_running(pid)) and time.time() < deadline:
        time.sleep(0.05)
    assert not child.alive()
    assert not _pid_running(pid)
    assert plugin.clients == []
    ctx.unmount_all()


def test_bridge_empty_servers_noop() -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    before = {t.name for t in ctx.require("tools").all()}
    ctx.mount(BridgeMcpPlugin(), {"mcp_servers": []})
    after = {t.name for t in ctx.require("tools").all()}
    assert before == after
    ctx.unmount_all()


@pytest.mark.parametrize(
    ("remote", "expect_cacheable"),
    [
        ("read_file", True),
        ("delete_path", False),
        ("send_mail", False),
    ],
)
def test_cacheable_param(remote: str, expect_cacheable: bool) -> None:
    assert (not is_write_tool(remote)) is expect_cacheable
