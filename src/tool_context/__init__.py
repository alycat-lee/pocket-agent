"""工具调用上下文：线程本地存储当前 session_id，供 workspace 工具解析路径。"""

from __future__ import annotations

import threading

_local = threading.local()


def set_session_id(session_id: str) -> None:
    _local.value = session_id


def get_session_id() -> str:
    return getattr(_local, "value", "default")
