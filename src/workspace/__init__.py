"""工作区管理：路径围栏与会话绑定。"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pocket_agent.config import load_settings, save_settings

Permission = Literal["readonly", "workspace", "full"]

_WORKSPACE_KEY = "workspaces"
_SESSION_BIND_KEY = "session_workspaces"


@dataclass
class Workspace:
    id: str
    name: str
    root: str  # 目录绝对路径


@dataclass
class WorkspaceManager:
    """工作区列表 + 会话绑定，持久化到 settings.json。"""

    settings_path: Path
    _workspaces: dict[str, Workspace] = field(default_factory=dict)
    _session_bindings: dict[str, str] = field(default_factory=dict)
    _readonly_roots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        data = load_settings(self.settings_path)
        raw = data.get(_WORKSPACE_KEY)
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                ws = Workspace(
                    id=str(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    root=str(item.get("root") or ""),
                )
                self._workspaces[ws.id] = ws
        if not self._workspaces:
            # 默认工作区
            default = Workspace(id="default", name="default", root=os.getcwd())
            self._workspaces["default"] = default
        bindings = data.get(_SESSION_BIND_KEY)
        if isinstance(bindings, dict):
            self._session_bindings = {
                str(k): str(v) for k, v in bindings.items()
            }

    def _save(self) -> None:
        data = load_settings(self.settings_path)
        data[_WORKSPACE_KEY] = [
            {"id": ws.id, "name": ws.name, "root": ws.root}
            for ws in self._workspaces.values()
        ]
        data[_SESSION_BIND_KEY] = dict(self._session_bindings)
        save_settings(data, self.settings_path)

    def list(self) -> list[Workspace]:
        return list(self._workspaces.values())

    def get(self, workspace_id: str) -> Workspace | None:
        return self._workspaces.get(workspace_id)

    def ensure(self, workspace_id: str, name: str, root: str) -> Workspace:
        """注册或刷新工作区（不强制写入 settings）。"""
        abs_root = str(Path(root).resolve())
        Path(abs_root).mkdir(parents=True, exist_ok=True)
        ws = Workspace(id=workspace_id, name=name, root=abs_root)
        self._workspaces[workspace_id] = ws
        return ws

    def add_readonly_root(self, root: str) -> None:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        if resolved not in self._readonly_roots:
            self._readonly_roots.append(resolved)

    def create(self, name: str, root: str) -> Workspace:
        wid = uuid.uuid4().hex[:8]
        abs_root = str(Path(root).resolve())
        ws = Workspace(id=wid, name=name, root=abs_root)
        self._workspaces[wid] = ws
        self._save()
        return ws

    def bind_session(self, session_id: str, workspace_id: str) -> Workspace:
        if workspace_id not in self._workspaces:
            raise KeyError(f"workspace not found: {workspace_id}")
        self._session_bindings[session_id] = workspace_id
        self._save()
        return self._workspaces[workspace_id]

    def get_session_workspace(self, session_id: str) -> Workspace:
        wid = self._session_bindings.get(session_id, "default")
        ws = self._workspaces.get(wid)
        if ws is None:
            ws = self._workspaces.get("default")
        if ws is None:
            ws = Workspace(id="default", name="default", root=os.getcwd())
        return ws

    def resolve_path(self, session_id: str, path: str) -> str:
        """相对路径解析到会话工作区根目录；绝对路径原样返回。"""
        p = Path(path)
        if p.is_absolute():
            return str(p.resolve())
        ws = self.get_session_workspace(session_id)
        return str((Path(ws.root) / path).resolve())

    def validate_path(
        self,
        session_id: str,
        abs_path: str,
        *,
        write: bool = False,
    ) -> bool:
        """校验绝对路径前缀在会话工作区内；shared 只读根允许读。"""
        ws = self.get_session_workspace(session_id)
        try:
            root = Path(ws.root).resolve()
            target = Path(abs_path).resolve()
        except OSError:
            return False
        try:
            target.relative_to(root)
            return True
        except ValueError:
            pass
        if write:
            return False
        for extra in self._readonly_roots:
            try:
                target.relative_to(extra)
                return True
            except ValueError:
                continue
        return False
