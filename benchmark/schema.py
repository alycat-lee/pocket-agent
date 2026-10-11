"""任务集 schema：文档 15 任务卡必填字段 + 扩展字段。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).resolve().parent / "tasks"

DIMENSIONS: tuple[str, ...] = ("d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9")
TRACKS: tuple[str, ...] = ("bare", "pocket", "cloud")

DIM_FILES: dict[str, str] = {
    "d1": "d1_format.json",
    "d2": "d2_memory.json",
    "d3": "d3_tools.json",
    "d4": "d4_planning.json",
    "d5": "d5_verify.json",
    "d6": "d6_longform.json",
    "d7": "d7_knowledge.json",
    "d8": "d8_reasoning.json",
    "d9": "d9_multiturn.json",
}

DIM_LABELS: dict[str, str] = {
    "d1": "D1 指令遵循与格式",
    "d2": "D2 上下文与记忆",
    "d3": "D3 工具使用",
    "d4": "D4 任务理解与规划",
    "d5": "D5 自我校验纠错",
    "d6": "D6 长程连贯生成",
    "d7": "D7 知识获取",
    "d8": "D8 推理深度",
    "d9": "D9 多轮状态",
}

_TASK_REQUIRED = ("id", "dimension", "input", "rubric")
_RUBRIC_REQUIRED = ("pass_criteria", "score_fields")


class TaskSchemaError(ValueError):
    """任务集不符合文档 15 任务卡格式。"""


def validate_task(task: Any, *, source: str = "") -> dict[str, Any]:
    prefix = f"{source}: " if source else ""
    if not isinstance(task, dict):
        raise TaskSchemaError(f"{prefix}task must be object")
    missing = [k for k in _TASK_REQUIRED if k not in task]
    if missing:
        raise TaskSchemaError(f"{prefix}missing {missing}")
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        raise TaskSchemaError(f"{prefix}id empty")
    dim = str(task.get("dimension") or "").strip()
    if dim not in DIMENSIONS:
        raise TaskSchemaError(f"{prefix}invalid dimension {dim!r}")
    if not str(task.get("input") or "").strip() and not task.get("turns"):
        raise TaskSchemaError(f"{prefix}input empty")
    rubric = task.get("rubric")
    if not isinstance(rubric, dict):
        raise TaskSchemaError(f"{prefix}rubric must be object")
    rub_missing = [k for k in _RUBRIC_REQUIRED if k not in rubric]
    if rub_missing:
        raise TaskSchemaError(f"{prefix}rubric missing {rub_missing}")
    criteria = rubric.get("pass_criteria")
    fields = rubric.get("score_fields")
    if not isinstance(criteria, list) or not all(isinstance(x, str) for x in criteria):
        raise TaskSchemaError(f"{prefix}pass_criteria must be string list")
    if not isinstance(fields, list) or not fields or not all(isinstance(x, str) for x in fields):
        raise TaskSchemaError(f"{prefix}score_fields must be non-empty string list")
    constraints = task.get("expected_constraints")
    if constraints is not None and not isinstance(constraints, dict):
        raise TaskSchemaError(f"{prefix}expected_constraints must be object")
    extra_turns = task.get("turns")
    if extra_turns is not None and (
        not isinstance(extra_turns, list) or not all(isinstance(x, str) for x in extra_turns)
    ):
        raise TaskSchemaError(f"{prefix}turns must be string list")
    return task


def validate_task_file(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise TaskSchemaError(f"{path.name}: expected non-empty array")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        task = validate_task(item, source=f"{path.name}[{i}]")
        tid = str(task["id"])
        if tid in seen:
            raise TaskSchemaError(f"{path.name}: duplicate id {tid!r}")
        seen.add(tid)
        out.append(task)
    return out


def load_dimension_tasks(dimension: str, *, tasks_dir: Path | None = None) -> list[dict[str, Any]]:
    dim = dimension.strip().lower()
    if dim not in DIM_FILES:
        raise TaskSchemaError(f"unknown dimension {dimension!r}")
    root = tasks_dir or TASKS_DIR
    return validate_task_file(root / DIM_FILES[dim])


def load_tasks(
    dimensions: list[str],
    *,
    tasks_dir: Path | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for dim in dimensions:
        tasks.extend(load_dimension_tasks(dim, tasks_dir=tasks_dir))
    return tasks


def task_turns(task: dict[str, Any]) -> list[str]:
    first = str(task.get("input") or "").strip()
    extra = task.get("turns")
    rest = [str(x).strip() for x in extra] if isinstance(extra, list) else []
    rest = [x for x in rest if x]
    if first:
        return [first, *rest]
    return rest
