"""客观指标全自动；rubric 主观项走云端 judge（1~5）。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.task import TaskCard, parse_total_chars
from pocket_agent.verifier import verify
from pocket_agent.writer import WriterMetrics

from benchmark.runners import TrackResult

JUDGE_PROMPT = """你是评测裁判。根据评分标准给回答打分，分数必须是 1 到 5 的整数。

评分标准：
1 分：完全不符合标准，答非所问或无法使用
2 分：少量沾边，关键要求未满足
3 分：基本可用，有明显缺漏
4 分：大部分符合，仅有小问题
5 分：完全符合评分标准

任务：{task_input}
评分标准条目：
{criteria}

字段：{field}

回答：
{answer}

只输出 JSON。"""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}

OBJECTIVE_FIELDS = frozenset(
    {
        "parse_ok",
        "format_ok",
        "needle_hit",
        "tool_used",
        "dup_call_rate",
        "char_ratio",
        "numeric_ok",
        "type_ok",
        "constraint_ok",
        "no_think_leak",
        "leak_caught",
        "contains",
    }
)
JUDGE_FIELDS = frozenset({"quality", "coherence", "helpfulness", "accuracy", "reasoning"})

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BRACE = re.compile(r"\{.*\}", re.DOTALL)
_METRICS = WriterMetrics()


def score_result(
    task: dict[str, Any],
    result: TrackResult,
    *,
    judge: LLMClientProtocol | None = None,
    judge_dir: Path | None = None,
) -> dict[str, Any]:
    """返回 {field: float|None, judge: [...]}。客观项确定性；跳过/失败记 0。"""
    fields = [str(x) for x in task.get("rubric", {}).get("score_fields") or []]
    scores: dict[str, float | None] = {}
    judge_rows: list[dict[str, Any]] = []
    if result.skipped:
        return {"fields": {f: None for f in fields}, "judge": [], "skipped": True}
    if result.error and not result.answer:
        return {"fields": {f: 0.0 for f in fields}, "judge": [], "skipped": False}
    for field in fields:
        if field in JUDGE_FIELDS:
            value, row = _judge_field(task, result, field, judge, judge_dir)
            scores[field] = value
            if row is not None:
                judge_rows.append(row)
        else:
            scores[field] = _objective(field, task, result)
    return {"fields": scores, "judge": judge_rows, "skipped": False}


def mean_score(field_scores: dict[str, float | None]) -> float | None:
    values = [v for v in field_scores.values() if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _objective(field: str, task: dict[str, Any], result: TrackResult) -> float:
    answer = result.answer or ""
    if field in ("parse_ok", "format_ok"):
        return _score_parse(task, answer)
    if field == "needle_hit":
        needle = str(task.get("needle") or "").strip()
        return 1.0 if needle and needle in answer else 0.0
    if field == "contains":
        expected = task.get("expected")
        if expected is None:
            return 0.0
        if isinstance(expected, dict):
            return 1.0 if all(str(v) in answer for v in expected.values()) else 0.0
        return 1.0 if str(expected) in answer else 0.0
    if field == "tool_used":
        return _score_tool_used(task, result)
    if field == "dup_call_rate":
        if int(result.extras.get("real_calls") or 0) < 1:
            return 0.0
        raw_dup = result.extras.get("dup_call_rate")
        rate = float(raw_dup) if raw_dup is not None else 1.0
        return 1.0 if rate == 0.0 else 0.0
    if field == "char_ratio":
        target = _expected_chars(task)
        if target <= 0:
            return 0.0
        ratio = len(answer) / target
        return 1.0 if ratio >= 0.60 else max(0.0, ratio / 0.60)
    if field == "numeric_ok":
        return 1.0 if not _METRICS.numeric_consistency(answer) else 0.0
    if field == "type_ok":
        return _score_type(task, result)
    if field == "constraint_ok":
        return _score_constraints(task, result)
    if field == "no_think_leak":
        card = _card_for_verify(task, result)
        return 0.0 if "思维链泄漏" in verify(answer, card) else 1.0
    if field == "leak_caught":
        return _score_leak_caught(answer, result)
    return 0.0


def _score_tool_used(task: dict[str, Any], result: TrackResult) -> float:
    calls = result.extras.get("tool_calls")
    if not isinstance(calls, list):
        calls = []
    names = [
        str(item.get("name") or "")
        for item in calls
        if isinstance(item, dict)
    ]
    forbidden = {str(x) for x in (task.get("forbidden_tools") or []) if x}
    if forbidden and any(name in forbidden for name in names):
        return 0.0
    expected = [str(x) for x in (task.get("expected_tools") or []) if x]
    if not expected:
        return 1.0 if int(result.extras.get("real_calls") or 0) >= 1 else 0.0
    for item in calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name not in expected:
            continue
        if item.get("from_cache") or not item.get("error"):
            return 1.0
    return 0.0


def _score_parse(task: dict[str, Any], answer: str) -> float:
    spec = task.get("parse") if isinstance(task.get("parse"), dict) else {}
    kind = str(spec.get("kind") or "json")
    text = (answer or "").strip()
    if kind == "json":
        obj = _extract_json(text)
        if obj is None:
            return 0.0
        keys = spec.get("keys") or []
        if isinstance(keys, list) and keys and not all(str(k) in obj for k in keys):
            return 0.5
        expected = task.get("expected")
        if isinstance(expected, dict):
            hits = sum(1 for k, v in expected.items() if obj.get(k) == v or str(obj.get(k)) == str(v))
            return 1.0 if hits == len(expected) else 0.5
        return 1.0
    if kind == "int":
        first = text.splitlines()[0].strip() if text else ""
        ok = bool(re.fullmatch(r"-?\d+", first))
        expected = task.get("expected")
        if ok and expected is not None and str(expected) not in first:
            return 0.5
        return 1.0 if ok else 0.0
    if kind == "number":
        return 1.0 if re.search(r"-?\d+(?:\.\d+)?", text) else 0.0
    if kind == "lines":
        count = int(spec.get("count") or 3)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) == count:
            return 1.0
        return 0.5 if abs(len(lines) - count) == 1 else 0.0
    if kind == "table":
        return 1.0 if "|" in text and "---" in text else 0.0
    if kind == "yaml_list":
        items = re.findall(r"^-\s+\S+", text, flags=re.MULTILINE)
        need = int(spec.get("count") or 1)
        return 1.0 if len(items) >= need else (0.5 if items else 0.0)
    return 0.0


def _extract_json(text: str) -> dict[str, Any] | None:
    block = _JSON_BLOCK.search(text)
    raw = block.group(1) if block else None
    if raw is None:
        brace = _JSON_BRACE.search(text)
        raw = brace.group(0) if brace else None
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _expected_chars(task: dict[str, Any]) -> int:
    constraints = task.get("expected_constraints")
    if isinstance(constraints, dict):
        raw = constraints.get("total_chars")
        if isinstance(raw, (int, float)) and int(raw) > 0:
            return int(raw)
    return parse_total_chars(str(task.get("input") or ""), default=0)


def _score_type(task: dict[str, Any], result: TrackResult) -> float:
    expected = (task.get("expected_constraints") or {}).get("type")
    if not expected:
        return 0.0
    card = result.extras.get("task_card")
    if isinstance(card, dict) and card.get("type") == expected:
        return 1.0
    text = result.answer or ""
    aliases = {
        "writer": ("writer", "小说", "长文", "翻译", "撰写"),
        "agent": ("agent", "工具", "天气", "查询"),
        "qa": ("qa", "闲聊", "问候"),
    }
    tokens = aliases.get(str(expected), (str(expected),))
    return 1.0 if any(tok in text for tok in tokens) else 0.0


def _score_constraints(task: dict[str, Any], result: TrackResult) -> float:
    expected = task.get("expected_constraints")
    if not isinstance(expected, dict):
        return 0.0
    want = expected.get("total_chars")
    if not isinstance(want, (int, float)):
        return 1.0
    want_i = int(want)
    card = result.extras.get("task_card")
    if isinstance(card, dict):
        got = card.get("constraints", {})
        if isinstance(got, dict) and int(got.get("total_chars") or 0) == want_i:
            return 1.0
        return 0.0
    text = result.answer or ""
    if str(want_i) in text or "2万" in text or "两万一" in text:
        return 1.0
    parsed = parse_total_chars(text, default=0)
    return 1.0 if parsed == want_i else 0.0


def _score_leak_caught(answer: str, result: TrackResult) -> float:
    reasons = result.extras.get("verify_reasons") or []
    if any("泄漏" in str(r) for r in reasons):
        return 1.0
    leaked = "可用工具" in answer or "当前进度" in answer
    return 0.0 if leaked else 1.0


def _card_for_verify(task: dict[str, Any], result: TrackResult) -> TaskCard:
    raw = result.extras.get("task_card")
    if isinstance(raw, dict) and raw.get("type") in ("qa", "agent", "writer"):
        return TaskCard(
            type=raw["type"],
            goal=str(raw.get("goal") or task.get("input") or ""),
            constraints=dict(raw.get("constraints") or {}),
            plan=list(raw.get("plan") or []),
            needs_history=bool(raw.get("needs_history")),
        )
    return TaskCard(type="qa", goal=str(task.get("input") or ""), constraints={}, plan=[])


def _judge_field(
    task: dict[str, Any],
    result: TrackResult,
    field: str,
    judge: LLMClientProtocol | None,
    judge_dir: Path | None,
) -> tuple[float | None, dict[str, Any] | None]:
    if judge is None:
        return None, {
            "task_id": result.task_id,
            "track": result.track,
            "field": field,
            "skipped": True,
            "reason": "judge 不可用",
        }
    criteria = task.get("rubric", {}).get("pass_criteria") or []
    prompt = JUDGE_PROMPT.format(
        task_input=task.get("input", ""),
        criteria="\n".join(f"- {c}" for c in criteria),
        field=field,
        answer=result.answer or "",
    )
    row: dict[str, Any] = {
        "task_id": result.task_id,
        "track": result.track,
        "field": field,
        "prompt": prompt,
    }
    try:
        data = judge.generate_json(prompt, JUDGE_SCHEMA)
        score = int(data.get("score", 0))
        rationale = str(data.get("rationale") or "")
        row["score"] = score
        row["rationale"] = rationale
        value = max(1, min(5, score)) / 5.0
    except (LLMFormatError, TypeError, ValueError) as exc:
        row["error"] = str(exc)
        value = None
    if judge_dir is not None:
        judge_dir.mkdir(parents=True, exist_ok=True)
        path = judge_dir / f"{result.track}_{result.task_id}_{field}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        row["path"] = str(path)
    return value, row
