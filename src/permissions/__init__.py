"""权限规则引擎：allow / ask / deny。无用户规则时回退 cacheable 语义。"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from typing import Any

_EFFECTS = frozenset({"allow", "ask", "deny"})


@dataclass
class PermissionRule:
    effect: str  # "allow" | "ask" | "deny"
    tool: str = "*"  # fnmatch 模式，匹配工具名
    pattern: str = ""  # 正则；空=不约束
    note: str = ""  # 拒绝/询问时展示给用户


def builtin_rules() -> list[PermissionRule]:
    """内置兜底：灾难命令 deny，其余 run_command / 写文件 ask。"""
    return [
        PermissionRule(
            effect="deny",
            tool="run_command",
            pattern=r"rm\s+-[a-z]*r[a-z]*f",
            note="拒绝递归强制删除（rm -rf）",
        ),
        PermissionRule(
            effect="deny",
            tool="run_command",
            pattern=r"del\s+/[fsq]",
            note="拒绝 Windows 强制删除（del /f /s /q）",
        ),
        PermissionRule(
            effect="deny",
            tool="run_command",
            pattern=r"format\s+[a-zA-Z]:",
            note="拒绝格式化磁盘",
        ),
        PermissionRule(
            effect="deny",
            tool="run_command",
            pattern=r":\(\)\s*\{",
            note="拒绝 fork 炸弹",
        ),
        PermissionRule(
            effect="deny",
            tool="run_command",
            pattern=r"mkfs",
            note="拒绝创建文件系统（mkfs）",
        ),
        PermissionRule(
            effect="deny",
            tool="run_command",
            pattern=r"dd\s+if=",
            note="拒绝磁盘覆写（dd if=）",
        ),
        PermissionRule(effect="ask", tool="run_command", note="执行命令需审批"),
        PermissionRule(effect="ask", tool="write_file", note="写入文件需审批"),
        PermissionRule(effect="ask", tool="edit_file", note="修改文件需审批"),
    ]


def parse_rule(raw: dict[str, Any] | PermissionRule) -> PermissionRule | None:
    if isinstance(raw, PermissionRule):
        if raw.effect not in _EFFECTS:
            return None
        return raw
    effect = str(raw.get("effect") or "")
    if effect not in _EFFECTS:
        return None
    return PermissionRule(
        effect=effect,
        tool=str(raw.get("tool") or "*"),
        pattern=str(raw.get("pattern") or ""),
        note=str(raw.get("note") or ""),
    )


def _pattern_hit(pattern: str, args: dict[str, Any]) -> bool:
    if not pattern:
        return True
    try:
        regex = re.compile(pattern)
    except re.error:
        return False
    blob = json.dumps(args, ensure_ascii=False)
    if regex.search(blob) is not None:
        return True
    # 命令白名单常写成 ^echo ，对 dumps 的 JSON 无法锚定，再匹配字符串参数
    for value in args.values():
        if isinstance(value, str) and regex.search(value) is not None:
            return True
    return False


def rule_matches(rule: PermissionRule, tool_name: str, args: dict[str, Any]) -> bool:
    if not fnmatch.fnmatch(tool_name, rule.tool or "*"):
        return False
    return _pattern_hit(rule.pattern, args)


def evaluate(
    user_rules: list[dict[str, Any]] | list[PermissionRule] | None,
    tool_name: str,
    args: dict[str, Any],
    *,
    cacheable: bool,
) -> tuple[str, PermissionRule]:
    """按用户规则再内置兜底顺序首个命中者生效；无命中回退 cacheable。"""
    chain: list[PermissionRule] = []
    for raw in user_rules or []:
        parsed = parse_rule(raw)
        if parsed is not None:
            chain.append(parsed)
    chain.extend(builtin_rules())
    for rule in chain:
        if rule_matches(rule, tool_name, args):
            return rule.effect, rule
    fallback = "allow" if cacheable else "ask"
    return fallback, PermissionRule(effect=fallback, tool=tool_name)
