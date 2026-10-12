"""Shell 插件：在会话工作区根执行命令。不受路径围栏约束。"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any, ClassVar

from pocket_agent.runtime import Context
from pocket_agent.tool_context import get_session_id
from pocket_agent.tools import ToolRegistry
from pocket_agent.workspace import WorkspaceManager

Disposer = Callable[[], None]


def _positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def run_command_in(
    command: str,
    *,
    cwd: str,
    timeout_sec: int,
    max_output_chars: int,
) -> str:
    """真实 subprocess；超时与超长输出返回错误/截断文本，非零退出码不抛。"""
    try:
        proc = subprocess.run(
            command,
            check=False,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return f"[工具错误] 命令超时（{timeout_sec}s），进程已终止"
    except OSError as exc:
        return f"[工具错误] {exc}"
    combined = (proc.stdout or "") + (proc.stderr or "")
    total = len(combined)
    if total > max_output_chars:
        combined = combined[:max_output_chars] + f"…（输出过长已截断，共 {total} 字符）"
    return f"退出码 {proc.returncode}\n{combined}"


def register_shell_tools(
    registry: ToolRegistry,
    manager: WorkspaceManager,
    *,
    timeout_sec: int,
    max_output_chars: int,
) -> list[Disposer]:
    def run_command(command: str) -> str:
        cwd = manager.resolve_path(get_session_id(), ".")
        return run_command_in(
            command,
            cwd=cwd,
            timeout_sec=timeout_sec,
            max_output_chars=max_output_chars,
        )

    return [
        registry.register_with_disposer(
            run_command,
            "在工作区执行命令",
            cacheable=False,
        ),
    ]


class ShellPlugin:
    """注册 run_command。防线 = 规则引擎 + 审批 + readonly 排除。"""

    name = "shell"
    requires: ClassVar[list[str]] = ["tools", "workspace"]
    provides: ClassVar[list[str]] = []

    def apply(self, ctx: Context, config: dict) -> None:
        registry: ToolRegistry = ctx.require("tools")
        manager: WorkspaceManager = ctx.require("workspace")
        timeout_sec = _positive_int(config.get("timeout_sec"), 60)
        max_output_chars = _positive_int(config.get("max_output_chars"), 4000)
        for disposer in register_shell_tools(
            registry,
            manager,
            timeout_sec=timeout_sec,
            max_output_chars=max_output_chars,
        ):
            ctx.on_dispose(disposer)
