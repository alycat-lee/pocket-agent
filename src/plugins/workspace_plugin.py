"""工作区插件：路径围栏 + 读/写/改/列/搜工具。"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from pocket_agent.runtime import Context
from pocket_agent.tool_context import get_session_id
from pocket_agent.tools import ToolRegistry
from pocket_agent.workspace import WorkspaceManager

Disposer = Callable[[], None]

_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__"})
_LIST_LIMIT = 200
_LIST_MAX_DEPTH = 2
_GREP_PER_FILE = 20
_GREP_TOTAL = 100
_GREP_MAX_BYTES = 1_000_000
_READ_LIMIT = 2000


def path_escape_error(path: str, ws_root: str) -> str:
    return f"[工具错误] 路径 {path} 超出工作区 {ws_root}，请使用工作区内相对路径"


def atomic_write_text(path: Path, content: str) -> None:
    """写 `<name>.tmp` 再 replace，避免中断留下半成品。"""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _ws_root(manager: WorkspaceManager, session_id: str) -> str:
    return manager.get_session_workspace(session_id).root


def _resolve_or_error(
    manager: WorkspaceManager,
    session_id: str,
    path: str,
    *,
    write: bool = False,
) -> Path | str:
    abs_path = manager.resolve_path(session_id, path)
    if not manager.validate_path(session_id, abs_path, write=write):
        return path_escape_error(path, _ws_root(manager, session_id))
    return Path(abs_path)


def fenced_read_file(manager: WorkspaceManager, session_id: str, path: str) -> str:
    resolved = _resolve_or_error(manager, session_id, path)
    if isinstance(resolved, str):
        return resolved
    if not resolved.is_file():
        return f"文件不存在：{path}"
    try:
        return resolved.read_text(encoding="utf-8")[:_READ_LIMIT]
    except OSError as exc:
        return f"读取失败：{exc}"


def fenced_write_file(
    manager: WorkspaceManager,
    session_id: str,
    path: str,
    content: str,
) -> str:
    resolved = _resolve_or_error(manager, session_id, path, write=True)
    if isinstance(resolved, str):
        return resolved
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(resolved, content)
        return f"已写入 {len(content)} 字符：{path}"
    except OSError as exc:
        return f"写入失败：{exc}"


def fenced_edit_file(
    manager: WorkspaceManager,
    session_id: str,
    path: str,
    old: str,
    new: str,
) -> str:
    resolved = _resolve_or_error(manager, session_id, path, write=True)
    if isinstance(resolved, str):
        return resolved
    if not resolved.is_file():
        return f"文件不存在：{path}"
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return f"读取失败：{exc}"
    n = text.count(old) if old else 0
    if n == 0:
        return "[工具错误] 未找到匹配文本，请先用 read_file 确认原文"
    if n > 1:
        return f"[工具错误] 匹配到 {n} 处，请在 old 中补充更多上下文使其唯一"
    try:
        atomic_write_text(resolved, text.replace(old, new, 1))
    except OSError as exc:
        return f"写入失败：{exc}"
    return f"已替换：{path}（{len(old)} → {len(new)} 字符）"


def fenced_list_dir(
    manager: WorkspaceManager,
    session_id: str,
    path: str = ".",
) -> str:
    resolved = _resolve_or_error(manager, session_id, path)
    if isinstance(resolved, str):
        return resolved
    if not resolved.is_dir():
        return f"不是目录：{path}"
    rows = _collect_list_rows(resolved)
    total = len(rows)
    shown = rows[:_LIST_LIMIT]
    lines = [item[1] for item in shown]
    if total > _LIST_LIMIT:
        lines.append(f"…（共 {total} 条，已截断）")
    return "\n".join(lines) if lines else "（空目录）"


def fenced_grep_files(
    manager: WorkspaceManager,
    session_id: str,
    pattern: str,
    path: str = ".",
) -> str:
    resolved = _resolve_or_error(manager, session_id, path)
    if isinstance(resolved, str):
        return resolved
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"[工具错误] 正则无效：{exc}"
    hits = _collect_grep_hits(resolved, regex)
    total = len(hits)
    shown = hits[:_GREP_TOTAL]
    if not shown:
        return "（无匹配）"
    if total > _GREP_TOTAL:
        shown.append(f"…（共 {total} 条，已截断）")
    return "\n".join(shown)


def _collect_list_rows(root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def walk(current: Path, depth: int) -> None:
        if depth > _LIST_MAX_DEPTH:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.name in _SKIP_DIRS and entry.is_dir():
                continue
            rel = entry.relative_to(root).as_posix()
            if entry.is_dir():
                rows.append((rel, f"[dir] {rel}/"))
                walk(entry, depth + 1)
            elif entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                rows.append((rel, f"[file] {rel} {size}B"))

    walk(root, 1)
    return rows


def _is_text_file(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > _GREP_MAX_BYTES:
        return False
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _collect_grep_hits(root: Path, regex: re.Pattern[str]) -> list[str]:
    hits: list[str] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                stack.append(entry)
                continue
            if not entry.is_file() or not _is_text_file(entry):
                continue
            try:
                text = entry.read_text(encoding="utf-8")
            except OSError:
                continue
            rel = entry.relative_to(root).as_posix()
            per_file = 0
            for index, line in enumerate(text.splitlines(), start=1):
                if regex.search(line) is None:
                    continue
                if per_file >= _GREP_PER_FILE:
                    break
                hits.append(f"{rel}:{index}: {line}")
                per_file += 1
    return hits


def register_workspace_tools(
    registry: ToolRegistry,
    manager: WorkspaceManager,
) -> list[Disposer]:
    """注册围栏版读写工具，返回 disposer。"""

    def read_file(path: str) -> str:
        return fenced_read_file(manager, get_session_id(), path)

    def write_file(path: str, content: str) -> str:
        return fenced_write_file(manager, get_session_id(), path, content)

    def edit_file(path: str, old: str, new: str) -> str:
        return fenced_edit_file(manager, get_session_id(), path, old, new)

    def list_dir(path: str = ".") -> str:
        return fenced_list_dir(manager, get_session_id(), path)

    def grep_files(pattern: str, path: str = ".") -> str:
        return fenced_grep_files(manager, get_session_id(), pattern, path)

    return [
        registry.register_with_disposer(read_file, "读取工作区内文件", cacheable=True),
        registry.register_with_disposer(write_file, "写入工作区内文件", cacheable=False),
        registry.register_with_disposer(edit_file, "替换文件中的文本", cacheable=False),
        registry.register_with_disposer(list_dir, "列出工作区目录", cacheable=True),
        registry.register_with_disposer(grep_files, "搜索工作区文本", cacheable=True),
    ]


class WorkspacePlugin:
    """工作区服务 → workspace（路径围栏 + 会话绑定）。"""

    name = "workspace"
    requires: ClassVar[list[str]] = ["tools"]
    provides: ClassVar[list[str]] = ["workspace"]

    def apply(self, ctx: Context, config: dict) -> None:
        settings_path = Path(str(config.get("settings_path") or "settings.json"))
        manager = WorkspaceManager(settings_path=settings_path)
        ctx.provide("workspace", manager)

        registry: ToolRegistry = ctx.require("tools")
        registry.unregister("read_file")
        for disposer in register_workspace_tools(registry, manager):
            ctx.on_dispose(disposer)
