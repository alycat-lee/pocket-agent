"""交付前纯代码验收：不调模型。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pocket_agent.prompts as prompt_mod
from pocket_agent.knowledge import EMPTY_RESULT, HONEST_PHRASES
from pocket_agent.prompt_registry import current_registry
from pocket_agent.prompts import builtin_entries
from pocket_agent.task import TaskCard

DEGRADE_PREFIX = "该回答未通过验收："

_THINK_TAG = re.compile(r"</?think\b", re.IGNORECASE)
_THINK_HEADS = ("嗯，", "我现在需要", "首先，我得")
_DATE_RE = re.compile(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?|\d{1,2}月\d{1,2}日")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_PROPER_RE = re.compile(
    r"《[^》]+》"
    r"|[A-Za-z][A-Za-z0-9_\-]{1,31}"
    r"|[\u4e00-\u9fff]{2,12}(?:协议|条约|公司|组织|计划|项目|手册|港口|币)"
)
_CLAIM_STOP = {
    "根据",
    "资料",
    "检索",
    "回答",
    "问题",
    "相关",
    "找到",
    "未找到",
    "这个",
    "一个",
    "我们",
}

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
    """从全部提示词变体筛出内部指令词。"""
    texts: list[str] = []
    registry = current_registry()
    if registry is not None:
        for record in registry.list():
            texts.extend(record.variants.values())
    else:
        for entry in builtin_entries():
            texts.extend(str(v) for v in dict(entry["variants"]).values())
    for name, value in vars(prompt_mod).items():
        if name.isupper() and isinstance(value, str):
            texts.append(str.__str__(value))
    blob = "\n".join(texts)
    return tuple(phrase for phrase in _LEAK_CANDIDATES if phrase in blob)


def verify(answer: str, task_card: TaskCard, *, evidence: str = "") -> list[str]:
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
    if task_card.needs_knowledge:
        problems.extend(_citation_problems(text, evidence))
    return problems


def extract_claims(text: str) -> list[str]:
    """抽取数字、日期、专名，供引用校验。"""
    body = text or ""
    claims: list[str] = []
    for match in _DATE_RE.finditer(body):
        claims.append(match.group(0))
    for match in _NUM_RE.finditer(body):
        token = match.group(0)
        if any(token in item for item in claims):
            continue
        claims.append(token)
    for match in _PROPER_RE.finditer(body):
        token = match.group(0).strip("《》")
        if token and token not in _CLAIM_STOP:
            claims.append(token)
    unique: list[str] = []
    for item in claims:
        if item not in unique:
            unique.append(item)
    return unique


def _citation_problems(answer: str, evidence: str) -> list[str]:
    ev = evidence or ""
    empty = (not ev.strip()) or EMPTY_RESULT in ev
    honest = any(phrase in (answer or "") for phrase in HONEST_PHRASES)
    if empty:
        if honest:
            return []
        return ["请仅基于检索资料回答"]
    claims = extract_claims(answer)
    if not claims:
        return []
    missing = [claim for claim in claims if claim not in ev]
    if len(missing) / len(claims) > 0.30:
        return ["请仅基于检索资料回答"]
    return []


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
