from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pocket_agent.runtime import Context
from pocket_agent.tool_context import get_session_id
from pocket_agent.tools import ToolRegistry
from pocket_agent.workspace import WorkspaceManager


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

        def read_file(path: str) -> str:
            sid = get_session_id()
            abs = manager.resolve_path(sid, path)
            if not manager.validate_path(sid, abs):
                return "[工具错误] 路径超出工作区范围"
            fp = Path(abs)
            if not fp.is_file():
                return f"文件不存在：{path}"
            try:
                return fp.read_text(encoding="utf-8")[:2000]
            except OSError as exc:
                return f"读取失败：{exc}"

        def write_file(path: str, content: str) -> str:
            sid = get_session_id()
            abs = manager.resolve_path(sid, path)
            if not manager.validate_path(sid, abs):
                return "[工具错误] 路径超出工作区范围"
            fp = Path(abs)
            try:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
                return f"已写入：{path}"
            except OSError as exc:
                return f"写入失败：{exc}"

        # 覆盖 tools-core 的 read_file（如果已注册则先注销）
        try:
            registry.unregister("read_file")
        except Exception:  # noqa: BLE001, S110
            pass
        registry.register(read_file, "读取工作区内文件", cacheable=True)
        registry.register(write_file, "写入工作区内文件", cacheable=False)
