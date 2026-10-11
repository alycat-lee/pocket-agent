"""交付前纯代码验收：不调模型。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pocket_agent.prompts as prompt_mod
from pocket_agent.task import TaskCard

DEGRADE_PREFIX = "该回答未通过验收："

_THINK_TAG = re.compile(r"</?think\b", re.IGNORECASE)
_THINK_HEADS = ("嗯，", "我现在需要", "首先，我得")
_LEAK_CANDIDATES = (
    "可用工具",
    "当前进度",
    "只输出",
    "已执行的历史",
    "不要输出任何解释",
    "基于以上信息回答任务",
    "不要复述过程",
    "下一步应该使用哪个工具",
    "分析用户任务",
    "只输出 JSON",
    "只输出一个数字",
    "只输出本段正文",
)


def prompt_leak_phrases() -> tuple[str, ...]:
    """从 prompts.py 模板自动筛出内部指令词。"""
    blob = "\n".join(
        str(value)
        for name, value in vars(prompt_mod).items()
        if name.isupper() and isinstance(value, str)
    )
    return tuple(phrase for phrase in _LEAK_CANDIDATES if phrase in blob)


def verify(answer: str, task_card: TaskCard) -> list[str]:
    """返回问题列表；空列表表示通过。"""
    text = answer or ""
    problems: list[str] = []
    if _think_leaked(text):
        problems.append("思维链泄漏")
    leaked = [p for p in prompt_leak_phrases() if _phrase_leaked(p, text)]
    if leaked:
        problems.append("prompt 泄漏")
    total = task_card.constraints.get("total_chars")
    if (
        isinstance(total, (int, float))
        and int(total) > 0
        and len(text) < int(int(total) * 0.60)
    ):
        problems.append("字数不足")
    if len(text.strip()) < 20 and task_card.type == "writer":
        problems.append("回答过短")
    if _parrot_plan(text, task_card.plan):
        problems.append("复读计划")
    return problems


def attach_degrade(answer: str, reasons: list[str]) -> str:
    note = DEGRADE_PREFIX + "；".join(reasons)
    body = (answer or "").rstrip()
    if not body:
        return note
    return f"{body}\n\n{note}"


def _phrase_leaked(phrase: str, text: str) -> bool:
    """短指令词「只输出」不匹配「只输出了/的/过」等叙事完成式。"""
    if phrase != "只输出":
        return phrase in text
    return re.search(r"只输出(?![了的过着])", text) is not None


def _think_leaked(text: str) -> bool:
    if _THINK_TAG.search(text):
        return True
    head = text.lstrip()[:24]
    return any(head.startswith(token) for token in _THINK_HEADS)


def _parrot_plan(answer: str, plan: list[str]) -> bool:
    joined = "".join(p.strip() for p in plan if p and str(p).strip())
    if not joined or not answer.strip():
        return False
    ratio = SequenceMatcher(None, joined, answer.strip()).ratio()
    return ratio > 0.80
