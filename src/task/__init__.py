"""任务卡与 Planner：一次 generate_json 产出结构化任务理解。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import PLANNER

TaskType = Literal["qa", "agent", "writer"]
KnowledgeMode = Literal["none", "augment", "strict"]

TASK_CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["qa", "agent", "writer"]},
        "goal": {"type": "string"},
        "constraints": {"type": "object"},
        "plan": {"type": "array", "items": {"type": "string"}},
        "needs_history": {"type": "boolean"},
        "needs_knowledge": {"type": "boolean"},
    },
    "required": ["type", "goal", "constraints", "plan", "needs_history"],
}

_TOTAL_CHARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万)?\s*字")
_MODES: tuple[TaskType, ...] = ("qa", "agent", "writer")
# deprecated：新逻辑一律走 infer_knowledge_mode；保留导出兼容。
_KNOWLEDGE_HINTS = (
    "是什么",
    "谁写",
    "谁是",
    "哪一年",
    "哪年",
    "何时",
    "根据文档",
    "知识库",
    "市值",
    "签署",
    "作者",
    "最新",
    "今年",
    "百科",
    "资料",
    "几月几日",
    "哪个国家",
)
_IDENTITY_HINTS = (
    "你是谁",
    "你是什么",
    "介绍一下你自己",
    "自我介绍",
    "你会什么",
    "你能做什么",
)
_STRICT_HINTS = (
    "根据文档",
    "按照文档",
    "根据资料",
    "按资料",
    "知识库",
    "资料库",
    "文档里",
    "文档中",
    "根据我上传",
    "参考资料回答",
)
_AUGMENT_HINTS = (
    "最新",
    "今年",
    "市值",
    "签署",
    "哪一年",
    "哪年",
    "何时",
    "几月几日",
    "哪个国家",
    "百科",
    "是什么",
    "谁是",
    "谁写",
    "作者",
)
_KNOWLEDGE_MODES: tuple[KnowledgeMode, ...] = ("none", "augment", "strict")
DEFAULT_WRITER_PLAN: list[str] = [
    "确定题材与人物",
    "展开开篇场景",
    "推进冲突",
    "推向高潮",
    "收束全文",
]


@dataclass
class TaskCard:
    type: TaskType
    goal: str
    constraints: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    needs_history: bool = False
    needs_knowledge: bool = False
    knowledge_mode: KnowledgeMode = "none"

    def __post_init__(self) -> None:
        if self.knowledge_mode not in _KNOWLEDGE_MODES:
            self.knowledge_mode = "none"
        if self.knowledge_mode == "none" and self.needs_knowledge:
            self.knowledge_mode = "strict"
        self.needs_knowledge = self.knowledge_mode != "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_knowledge_mode(text: str) -> KnowledgeMode:
    """三档知识依赖：身份 none / 显式资料 strict / 事实 augment。"""
    body = text or ""
    if any(token in body for token in _IDENTITY_HINTS):
        return "none"
    if any(token in body for token in _STRICT_HINTS):
        return "strict"
    if any(token in body for token in _AUGMENT_HINTS):
        return "augment"
    return "none"


def infer_needs_knowledge(text: str) -> bool:
    """deprecated：请使用 infer_knowledge_mode。"""
    return infer_knowledge_mode(text) != "none"


def parse_total_chars(text: str, default: int = 0) -> int:
    """解析「N万字 / N字」；未命中返回 default（0 表示无约束）。"""
    match = _TOTAL_CHARS_RE.search(text or "")
    if match is None:
        return max(0, default)
    n = float(match.group(1))
    if match.group(2):
        n *= 10000.0
    return max(1, int(n))


def default_card(mode: TaskType, goal: str, *, user_input: str = "") -> TaskCard:
    """兜底任务卡：规则/默认模式 + 从原文补字数约束。"""
    constraints: dict[str, Any] = {}
    chars = parse_total_chars(user_input or goal, default=0)
    if chars:
        constraints["total_chars"] = chars
    kmode = infer_knowledge_mode(user_input or goal)
    return TaskCard(
        type=mode,
        goal=goal or user_input,
        constraints=constraints,
        plan=list(DEFAULT_WRITER_PLAN) if mode == "writer" else [],
        needs_history=mode == "agent",
        needs_knowledge=kmode != "none",
        knowledge_mode=kmode,
    )


def card_from_dict(data: dict[str, Any], *, user_input: str = "") -> TaskCard:
    if not isinstance(data, dict):
        raise LLMFormatError("task card is not object")
    raw_type = str(data.get("type", "")).strip()
    if raw_type not in _MODES:
        raise LLMFormatError(f"invalid task type: {raw_type!r}")
    goal = str(data.get("goal") or user_input or "").strip()
    if not goal:
        raise LLMFormatError("task card goal empty")
    raw_constraints = data.get("constraints")
    constraints = dict(raw_constraints) if isinstance(raw_constraints, dict) else {}
    constraints = _normalize_constraints(constraints, user_input or goal)
    raw_plan = data.get("plan")
    plan: list[str] = []
    if isinstance(raw_plan, list):
        plan = [str(p).strip() for p in raw_plan if str(p).strip()]
    needs = bool(data.get("needs_history", False))
    raw_mode = str(data.get("knowledge_mode") or "").strip()
    if raw_mode in _KNOWLEDGE_MODES:
        kmode: KnowledgeMode = raw_mode  # type: ignore[assignment]
    elif "needs_knowledge" in data:
        kmode = "strict" if bool(data.get("needs_knowledge")) else "none"
    else:
        kmode = infer_knowledge_mode(user_input or goal)
    return TaskCard(
        type=raw_type,  # type: ignore[arg-type]
        goal=goal,
        constraints=constraints,
        plan=plan,
        needs_history=needs,
        needs_knowledge=kmode != "none",
        knowledge_mode=kmode,
    )


def _normalize_constraints(
    constraints: dict[str, Any],
    user_input: str,
) -> dict[str, Any]:
    """字数只认原文（或模型返回的「N万字」字符串），丢弃臆造整数。"""
    out = dict(constraints)
    user_parsed = parse_total_chars(user_input, default=0)
    if user_parsed > 0:
        out["total_chars"] = user_parsed
        return out
    raw = out.get("total_chars")
    if isinstance(raw, str):
        parsed = parse_total_chars(raw, default=0)
        if parsed > 0:
            out["total_chars"] = parsed
            return out
    out.pop("total_chars", None)
    return out


class Planner:
    """一次 generate_json → TaskCard。"""

    def __init__(self, llm: LLMClientProtocol) -> None:
        self._llm = llm

    def classify(self, user_input: str, rolling_summary: str = "") -> TaskCard:
        return self.plan(user_input, rolling_summary)

    def plan(
        self,
        user_input: str,
        rolling_summary: str = "",
        *,
        system: str = "",
    ) -> TaskCard:
        prompt = PLANNER.format(
            rolling_summary=(rolling_summary or "").strip() or "（无）",
            user_input=user_input,
        )
        if system.strip():
            prompt = f"{system.strip()}\n\n{prompt}"
        data = self._llm.generate_json(prompt, TASK_CARD_SCHEMA)
        if not isinstance(data, dict):
            raise LLMFormatError("planner output is not object")
        return card_from_dict(data, user_input=user_input)
