"""交付前纯代码验收：不调模型。"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import pocket_agent.prompts as prompt_mod
from pocket_agent.intent_router import has_tool_hint
from pocket_agent.knowledge import EMPTY_RESULT, HONEST_PHRASES
from pocket_agent.llm_client import EMPTY_THINK_NOTE
from pocket_agent.prompt_registry import current_registry
from pocket_agent.prompts import builtin_entries
from pocket_agent.task import TaskCard
from pocket_agent.tools import detect_tool_domains, tool_name_domain

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

_STEP_LOG_LINE = re.compile(r"^\d+\.\s+[A-Za-z_][A-Za-z0-9_]*")
_NUMBERED_ACTION = re.compile(
    r"^\s*\d+[.、]\s*.*(工具|调用|生成|读取|保存到)",
    re.MULTILINE,
)
_TRUNCATED_TAIL = re.compile(r"[，,:：]\s*$")
_PLACEHOLDER_LEAK = re.compile(r"（此处|\(此处|可能需要更多的信息")
_JSON_FENCE = re.compile(
    r"^```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_JSON_BLOCK = re.compile(r"```json\s*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBERED_ITEM = re.compile(r"^\s*\d+[.、]", re.MULTILINE)
_INT_ONLY = re.compile(r"\s*-?\d+\s*")
_EMPTY_LINK = re.compile(r"\[\]\([^)]*\)")
_ANSWER_DOMAIN_ORDER = ("weather", "time", "web")
_ANSWER_DOMAINS = frozenset(_ANSWER_DOMAIN_ORDER)
_DOMAIN_LABEL = {"weather": "天气", "time": "时间", "web": "网页"}
_INTERNAL_LEAKS = (
    "format_error",
    "cannot parse int",
    "invalid json",
    "[工具错误]",
    "[格式错误]",
    "generate_int",
    "generate_json",
)

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


def strip_empty_links(text: str) -> str:
    """去掉空标签 Markdown 残片 []() / [](url)，保留 [text](url)。"""
    return _EMPTY_LINK.sub("", text or "")


def verify(
    answer: str,
    task_card: TaskCard,
    *,
    evidence: str = "",
    user_input: str = "",
    real_calls: int | None = None,
    tool_names: list[str] | None = None,
) -> list[str]:
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
    if _parrot_plan(text, task_card.plan) or _json_exec_plan(text):
        problems.append("复读计划")
    if task_card.needs_knowledge and (
        getattr(task_card, "knowledge_mode", "none") == "strict"
        or (evidence and EMPTY_RESULT not in evidence and evidence.strip())
    ):
        problems.extend(_citation_problems(text, evidence))
    if (
        task_card.type == "agent"
        and real_calls is not None
        and real_calls == 0
        and has_tool_hint(user_input)
    ):
        problems.append("应调用工具但未调用")
    if EMPTY_THINK_NOTE in text or "模型仅输出了思考过程" in text:
        problems.append("正式回答为空")
    if _execution_log_leaked(text):
        problems.append("执行日志泄漏")
    if _format_violation(text, user_input):
        problems.append("格式违反")
    missing = _missing_domain_tool(
        user_input,
        card_type=task_card.type,
        real_calls=real_calls,
        tool_names=tool_names,
    )
    if missing:
        problems.append(missing)
    if _hallucinated_file_ref(text):
        problems.append("幻觉文件引用")
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


def honest_missing_data_note(user_input: str, reasons: list[str]) -> str | None:
    """缺工具数据时的上屏说明；None 表示不是这类失败。"""
    domain_reason = next((item for item in reasons if item.startswith("领域工具缺失")), "")
    idle = "应调用工具但未调用" in reasons
    if not domain_reason and not idle:
        return None
    if domain_reason.startswith("领域工具缺失：") and domain_reason.endswith("工具"):
        labels = domain_reason[len("领域工具缺失：") : -len("工具")]
    else:
        needed = detect_tool_domains(user_input) & _ANSWER_DOMAINS
        labels = "、".join(_DOMAIN_LABEL[d] for d in _ANSWER_DOMAIN_ORDER if d in needed)
    if not labels:
        labels = "对应"
    return f"未能获取实时数据（未调用{labels}工具），以上回答不保证准确。"


def _execution_log_leaked(text: str) -> bool:
    body = text or ""
    hay = body.lower()
    if any(token in body or token in hay for token in _INTERNAL_LEAKS):
        return True
    if "[缓存]" in body or "args={" in body:
        return True
    streak = 0
    for line in body.splitlines():
        if _STEP_LOG_LINE.match(line.strip()):
            streak += 1
            if streak >= 2:
                return True
        elif line.strip():
            streak = 0
    if len(_NUMBERED_ACTION.findall(body)) >= 2:
        return True
    if _TRUNCATED_TAIL.search(body):
        return True
    if _PLACEHOLDER_LEAK.search(body):
        return True
    return _truncated_json(body)


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


def _collect_keys(obj: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        keys.update(str(k) for k in obj)
        for value in obj.values():
            keys.update(_collect_keys(value))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_keys(item))
    return keys


def _json_exec_plan(text: str) -> bool:
    for match in _JSON_BLOCK.finditer(text or ""):
        try:
            data = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        keys = _collect_keys(data)
        if "step" in keys and ("tool" in keys or "args" in keys):
            return True
    return False


def _unbalanced_brackets(text: str) -> bool:
    curly = 0
    square = 0
    for char in text:
        if char == "{":
            curly += 1
        elif char == "}":
            curly -= 1
        elif char == "[":
            square += 1
        elif char == "]":
            square -= 1
        if curly < 0 or square < 0:
            return True
    return curly != 0 or square != 0


def _truncated_json(text: str) -> bool:
    body = text or ""
    if "```json" in body.lower() and _unbalanced_brackets(body):
        return True
    tail = body[-20:]
    return '"step":' in tail or '"args":' in tail


def _is_pure_json(text: str) -> bool:
    body = (text or "").strip()
    fenced = _JSON_FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def _missing_domain_tool(
    user_input: str,
    *,
    card_type: str,
    real_calls: int | None,
    tool_names: list[str] | None,
) -> str | None:
    if card_type != "agent" or real_calls is None or real_calls <= 0:
        return None
    if tool_names is None:
        return None
    needed = detect_tool_domains(user_input) & _ANSWER_DOMAINS
    if not needed:
        return None
    have = {tool_name_domain(name) for name in tool_names}
    missing = [d for d in _ANSWER_DOMAIN_ORDER if d in needed and d not in have]
    if not missing:
        return None
    labels = "、".join(_DOMAIN_LABEL[d] for d in missing)
    return f"领域工具缺失：{labels}工具"


def _hallucinated_file_ref(text: str) -> bool:
    body = text or ""
    for match in re.finditer("缺少", body):
        if "文件" in body[match.start() : match.start() + 20]:
            return True
    return False


def _format_violation(answer: str, user_input: str) -> bool:
    req = user_input or ""
    text = answer or ""
    if any(token in req for token in ("只输出整数", "只输出数字", "只答数字")):
        return _INT_ONLY.fullmatch(text) is None
    if "只输出 JSON" in req or "只输出JSON对象" in req:
        return not _is_pure_json(text)
    if "只回答" in req or "只输出" in req:
        body = text.strip()
        if len(body) > 30:
            return True
        if "\n-" in text or "**" in text:
            return True
        return _NUMBERED_ITEM.search(text) is not None
    return False
