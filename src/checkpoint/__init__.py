from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class CheckpointStore:
    """JSON 落盘到 .checkpoints/{run_id}.json"""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else Path(".checkpoints")

    def save(self, run_id: str, state: dict) -> None:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def load(self, run_id: str) -> dict | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            data: Any = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if not all(key in data for key in ("history", "ledger_records", "step_index")):
            return None
        if not isinstance(data["history"], list):
            return None
        if not isinstance(data["ledger_records"], dict):
            return None
        if not isinstance(data["step_index"], int):
            return None
        return data

    def clear(self, run_id: str) -> None:
        path = self._path(run_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return
        tmp = path.with_suffix(".tmp")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            return

    def _path(self, run_id: str) -> Path:
        safe = re.sub(r"[^\w.\-]+", "_", run_id.strip()) or "default"
        return self._root / f"{safe}.json"
