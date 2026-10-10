from __future__ import annotations

from dataclasses import fields
from typing import Any

from pocket_agent.config import AgentConfig


def agent_config_from(data: dict[str, Any] | None = None) -> AgentConfig:
    """从插件 config 字典构造 AgentConfig（忽略未知键）。"""
    raw = dict(data or {})
    known = {f.name for f in fields(AgentConfig)}
    return AgentConfig(**{k: v for k, v in raw.items() if k in known})
