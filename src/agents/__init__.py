"""Agent 目录体系：目录即真相源。"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_AGENT = "default"
SHARED_NAME = "shared"
MEMBERS = ("agent.yml", "prompt.md", "memory", "skills", "sessions", "workspace")
LEGACY_SESSIONS = Path(".sessions")
LEGACY_MIGRATED = Path(".sessions.migrated")

_RESERVED = frozenset(
    {
        SHARED_NAME,
        ".",
        "..",
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "lpt1",
    }
)
_ILLEGAL_CHARS = set('<>:"|?*\0')
_NAME_OK = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff.\- ]{0,62}$")

DEFAULT_PROMPT = "你是 pocket-agent 的默认助手「default」，运行在用户本地的模型上。你可以直接回答问题、调用工具完成任务、生成长文。回答直接简洁，不编造来源。\n"


class AgentNameError(ValueError):
    """非法 agent 目录名（路径注入或保留名）。"""


@dataclass
class AgentProfile:
    name: str
    description: str = ""
    provider_id: str | None = None
    tools: list[str] | None = None
    permission: str = "workspace"
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("deleted", None)
        return data


@dataclass
class Skill:
    id: str
    name: str
    body: str
    keywords: list[str] = field(default_factory=list)


def validate_agent_name(name: str) -> str:
    """合法目录名；拒绝 ../、绝对路径、保留名。"""
    raw = (name or "").strip()
    if not raw:
        raise AgentNameError("agent 名称不能为空")
    lowered = raw.lower()
    if lowered in _RESERVED:
        raise AgentNameError(f"保留名称不可用：{raw}")
    if any(ch in raw for ch in _ILLEGAL_CHARS):
        raise AgentNameError("非法目录名")
    if any(sep in raw for sep in ("/", "\\")):
        raise AgentNameError("非法目录名")
    if ".." in raw or raw.startswith("."):
        raise AgentNameError("非法目录名")
    if Path(raw).is_absolute() or Path(raw).name != raw:
        raise AgentNameError("非法目录名")
    if _NAME_OK.fullmatch(raw) is None:
        raise AgentNameError("非法目录名")
    return raw


class AgentService:
    """agents/ 根目录上的 CRUD 与记忆/技能装配。"""

    def __init__(
        self,
        root: str | Path = "agents",
        *,
        legacy_sessions: Path | None = None,
    ) -> None:
        self._root = Path(root)
        self._legacy = Path(legacy_sessions) if legacy_sessions is not None else LEGACY_SESSIONS
        self.ensure_tree()
        self.migrate_legacy_sessions()

    def root(self) -> Path:
        return self._root

    def shared_dir(self) -> Path:
        return self._root / SHARED_NAME

    def set_root(self, path: str | Path) -> None:
        """改根目录只影响此后新建，不搬迁已有 agents。"""
        self._root = Path(path)
        self.ensure_tree()

    def ensure_tree(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        shared = self.shared_dir()
        for child in ("memory", "sessions", "files"):
            (shared / child).mkdir(parents=True, exist_ok=True)
        shared_entities = shared / "memory" / "entities.json"
        if not shared_entities.is_file():
            shared_entities.write_text("{}\n", encoding="utf-8")
        if not (self._root / DEFAULT_AGENT / "agent.yml").is_file():
            self._scaffold(
                DEFAULT_AGENT,
                description="默认助手",
            )

    def create(
        self,
        name: str,
        *,
        description: str = "",
        provider_id: str | None = None,
        tools: list[str] | None = None,
        permission: str = "workspace",
        prompt: str = "",
    ) -> AgentProfile:
        safe = validate_agent_name(name)
        yml = self.agent_dir(safe) / "agent.yml"
        if yml.is_file():
            existing = self._read_profile(safe)
            if not existing.deleted:
                raise ValueError(f"agent 已存在：{safe}")
        self._scaffold(
            safe,
            description=description,
            provider_id=provider_id,
            tools=tools,
            permission=permission,
            prompt=prompt,
        )
        return self.get(safe)

    def list(self) -> list[AgentProfile]:
        rows: list[AgentProfile] = []
        if not self._root.is_dir():
            return rows
        for path in sorted(self._root.iterdir(), key=lambda p: (p.name != DEFAULT_AGENT, p.name)):
            if not path.is_dir() or path.name == SHARED_NAME:
                continue
            try:
                validate_agent_name(path.name)
            except AgentNameError:
                continue
            if not (path / "agent.yml").is_file():
                continue
            profile = self._read_profile(path.name)
            if profile.deleted:
                continue
            rows.append(profile)
        return rows

    def get(self, name: str) -> AgentProfile:
        safe = validate_agent_name(name)
        if not (self.agent_dir(safe) / "agent.yml").is_file():
            raise KeyError(safe)
        profile = self._read_profile(safe)
        if profile.deleted:
            raise KeyError(safe)
        return profile

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        provider_id: str | None = None,
        tools: list[str] | None = None,
        permission: str | None = None,
        prompt: str | None = None,
    ) -> AgentProfile:
        profile = self.get(name)
        if description is not None:
            profile.description = description
        if provider_id is not None:
            profile.provider_id = provider_id or None
        if tools is not None:
            profile.tools = tools
        if permission is not None:
            profile.permission = permission or "workspace"
        self._write_profile(profile)
        if prompt is not None:
            self.write_prompt(profile.name, prompt)
        return profile

    def delete(self, name: str, keep_files: bool = True) -> None:
        safe = validate_agent_name(name)
        if safe == DEFAULT_AGENT:
            raise ValueError("不能删除 default agent")
        path = self.agent_dir(safe)
        if not path.is_dir():
            raise KeyError(safe)
        if keep_files:
            profile = self._read_profile(safe)
            profile.deleted = True
            self._write_profile(profile)
            return
        shutil.rmtree(path)

    def agent_dir(self, name: str) -> Path:
        return self._root / validate_agent_name(name)

    def sessions_dir(self, name: str) -> Path:
        if name == SHARED_NAME:
            return self.shared_dir() / "sessions"
        path = self.agent_dir(name) / "sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_dir(self, name: str) -> Path:
        path = self.agent_dir(name) / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_prompt(self, name: str) -> str:
        path = self.agent_dir(name) / "prompt.md"
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write_prompt(self, name: str, text: str) -> None:
        path = self.agent_dir(name) / "prompt.md"
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

    def read_entities(self, name: str) -> dict[str, str]:
        return _load_entity_map(self.agent_dir(name) / "memory" / "entities.json")

    def write_entities(self, name: str, data: dict[str, Any]) -> None:
        path = self.agent_dir(name) / "memory" / "entities.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({str(k): str(v) for k, v in data.items()}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def read_shared_entities(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        memory = self.shared_dir() / "memory"
        if not memory.is_dir():
            return merged
        files = sorted(memory.glob("*.json"))
        primary = memory / "entities.json"
        if primary.is_file():
            merged.update(_load_entity_map(primary))
        for path in files:
            if path.name == "entities.json":
                continue
            merged.update(_load_entity_map(path))
        return merged

    def write_shared_entities(self, data: dict[str, Any]) -> None:
        path = self.shared_dir() / "memory" / "entities.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({str(k): str(v) for k, v in data.items()}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def merge_entities(self, name: str) -> dict[str, str]:
        """共享只读 + 私有覆盖。"""
        merged = dict(self.read_shared_entities())
        merged.update(self.read_entities(name))
        return merged

    def format_entities(self, name: str) -> str:
        items = self.merge_entities(name)
        if not items:
            return ""
        lines = [f"{key}：{value}" for key, value in items.items()]
        return "已知实体：\n" + "\n".join(lines)

    def list_skills(self, name: str) -> list[Skill]:
        root = self.agent_dir(name) / "skills"
        if not root.is_dir():
            return []
        skills: list[Skill] = []
        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md"
            if not child.is_dir() or not skill_md.is_file():
                continue
            body = skill_md.read_text(encoding="utf-8")
            title = _skill_title(body) or child.name
            skills.append(
                Skill(
                    id=child.name,
                    name=title,
                    body=body,
                    keywords=_skill_keywords(body, child.name, title),
                )
            )
        return skills

    def match_skills(self, name: str, text: str) -> list[Skill]:
        blob = (text or "").lower()
        hits: list[Skill] = []
        for skill in self.list_skills(name):
            tokens = [skill.id.lower(), skill.name.lower(), *skill.keywords]
            if any(token and token in blob for token in tokens):
                hits.append(skill)
        return hits

    def system_preamble(self, name: str, task_text: str = "") -> str:
        """prompt.md + 实体槽位 + 命中技能，注入 Planner / final_answer。"""
        parts: list[str] = []
        try:
            profile = self.get(name)
            desc = profile.description or "运行在用户本地的 AI 助手"
        except (KeyError, AgentNameError):
            desc = "运行在用户本地的 AI 助手"
        parts.append(
            f"你是「{name}」，{desc}。基于本地模型运行，回答直接简洁。"
        )
        prompt = self.read_prompt(name).strip()
        if prompt:
            parts.append(prompt)
        entities = self.format_entities(name)
        if entities:
            parts.append(entities)
        for skill in self.match_skills(name, task_text):
            parts.append(f"技能「{skill.name}」：\n{skill.body.strip()}")
        return "\n\n".join(parts)

    def migrate_legacy_sessions(self, legacy: Path | None = None) -> int:
        """把旧 .sessions/*.json 迁到 agents/default/sessions/，幂等。"""
        src = Path(legacy) if legacy is not None else self._legacy
        dest = self.sessions_dir(DEFAULT_AGENT)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        if not src.is_dir():
            return copied
        for path in src.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            target = dest / path.name
            if not target.exists():
                shutil.copy2(path, target)
                copied += 1
        migrated = src.with_name(src.name + ".migrated") if src.name == ".sessions" else LEGACY_MIGRATED
        if src.resolve() == migrated.resolve():
            return copied
        if not migrated.exists():
            try:
                src.rename(migrated)
            except OSError:
                logger.warning("无法重命名 %s → %s", src, migrated)
        else:
            shutil.rmtree(src, ignore_errors=True)
        return copied

    def _scaffold(
        self,
        name: str,
        *,
        description: str = "",
        provider_id: str | None = None,
        tools: list[str] | None = None,
        permission: str = "workspace",
        prompt: str = "",
    ) -> None:
        root = self.agent_dir(name)
        (root / "memory").mkdir(parents=True, exist_ok=True)
        (root / "skills").mkdir(parents=True, exist_ok=True)
        (root / "sessions").mkdir(parents=True, exist_ok=True)
        (root / "workspace").mkdir(parents=True, exist_ok=True)
        profile = AgentProfile(
            name=name,
            description=description,
            provider_id=provider_id,
            tools=list(tools) if tools else None,
            permission=permission or "workspace",
            deleted=False,
        )
        self._write_profile(profile)
        prompt_path = root / "prompt.md"
        if not prompt_path.is_file() or prompt:
            prompt_path.write_text(prompt or DEFAULT_PROMPT, encoding="utf-8")
        entities = root / "memory" / "entities.json"
        if not entities.is_file():
            entities.write_text("{}\n", encoding="utf-8")

    def _read_profile(self, name: str) -> AgentProfile:
        raw = yaml.safe_load((self.agent_dir(name) / "agent.yml").read_text(encoding="utf-8"))
        data = raw if isinstance(raw, dict) else {}
        tools = data.get("tools")
        tool_list = [str(t) for t in tools] if isinstance(tools, list) else None
        provider = data.get("provider_id")
        return AgentProfile(
            name=str(data.get("name") or name),
            description=str(data.get("description") or ""),
            provider_id=str(provider) if provider else None,
            tools=tool_list,
            permission=str(data.get("permission") or "workspace"),
            deleted=bool(data.get("deleted", False)),
        )

    def _write_profile(self, profile: AgentProfile) -> None:
        path = self.agent_dir(profile.name) / "agent.yml"
        payload: dict[str, Any] = {
            "name": profile.name,
            "description": profile.description,
            "provider_id": profile.provider_id,
            "tools": profile.tools,
            "permission": profile.permission,
        }
        if profile.deleted:
            payload["deleted"] = True
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def _load_entity_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if str(k).strip()}


def _skill_title(body: str) -> str:
    for line in body.splitlines():
        text = line.strip()
        if text.startswith("#"):
            return text.lstrip("#").strip()
        if text:
            return text[:40]
    return ""


def _skill_keywords(body: str, dirname: str, title: str) -> list[str]:
    words = {dirname.lower(), title.lower()}
    for line in (body or "").splitlines()[:12]:
        if line.lower().startswith("keywords:"):
            extra = line.split(":", 1)[1]
            words.update(part.strip().lower() for part in extra.split(",") if part.strip())
    return [item for item in words if len(item) >= 2]


__all__ = [
    "DEFAULT_AGENT",
    "MEMBERS",
    "AgentNameError",
    "AgentProfile",
    "AgentService",
    "Skill",
    "validate_agent_name",
]
