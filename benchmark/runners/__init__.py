"""三跑道执行器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TrackResult:
    track: str
    task_id: str
    dimension: str
    answer: str
    turn_answers: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""


class TrackRunner(Protocol):
    name: str

    def run_task(self, task: dict[str, Any]) -> TrackResult: ...


def plan_only_fields(task: dict[str, Any]) -> bool:
    fields = task.get("rubric", {}).get("score_fields") or []
    return bool(fields) and set(fields) <= {"type_ok", "constraint_ok"}
