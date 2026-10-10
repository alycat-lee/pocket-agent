from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any


class EventStore:
    """会话事件落盘（JSONL），复用 .checkpoints/events/ 目录。

    每条记录：{id, ts, event, data}
    分页：从尾部向前，before=cursor（不含），limit=N。
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else Path(".checkpoints") / "events"
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}

    def append(
        self,
        session_id: str,
        event: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        sid = _safe_id(session_id)
        with self._lock:
            next_id = self._peek_next_id(sid)
            record = {
                "id": next_id,
                "ts": time.time(),
                "event": event,
                "data": data,
            }
            path = self._path(sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._seq[sid] = next_id
            return record

    def list_before(
        self,
        session_id: str,
        *,
        before: int | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """从尾部向前取 limit 条 id < before 的事件（按 id 升序返回）。

        返回 {events, next_before}；next_before 为下一页游标，无更早则为 null。
        """
        limit = max(limit, 0)
        sid = _safe_id(session_id)
        records = self._load_all(sid)
        if before is not None:
            records = [r for r in records if int(r["id"]) < int(before)]
        # 从尾部取 limit 条
        page = records[-limit:] if limit else []
        next_before: int | None = None
        if page and records:
            first_id = int(page[0]["id"])
            earlier = [r for r in records if int(r["id"]) < first_id]
            if earlier:
                next_before = first_id
        return {"events": page, "next_before": next_before}

    def export_all(self, session_id: str) -> list[dict[str, Any]]:
        return self._load_all(_safe_id(session_id))

    def clear(self, session_id: str) -> None:
        sid = _safe_id(session_id)
        path = self._path(sid)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        with self._lock:
            self._seq.pop(sid, None)

    def _peek_next_id(self, sid: str) -> int:
        if sid in self._seq:
            return self._seq[sid] + 1
        records = self._load_all_unlocked(sid)
        if not records:
            return 1
        return int(records[-1]["id"]) + 1

    def _load_all(self, sid: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._load_all_unlocked(sid)

    def _load_all_unlocked(self, sid: str) -> list[dict[str, Any]]:
        path = self._path(sid)
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "id" in item and "event" in item:
                out.append(item)
        out.sort(key=lambda r: int(r["id"]))
        if out:
            self._seq[sid] = int(out[-1]["id"])
        return out

    def _path(self, sid: str) -> Path:
        return self._root / f"{sid}.jsonl"


def _safe_id(session_id: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", session_id.strip()) or "default"
    return safe
