from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


def canonical_json(args: dict) -> str:
    """键排序、去空白的紧凑 JSON，保证语义相同参数生成相同指纹。"""
    return json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass
class CallLedger:
    """记录所有工具调用的指纹与结果，提供精确去重与缓存。"""

    _records: dict[str, str] = field(default_factory=dict)
    real_call_count: int = 0  # 真实执行次数
    hit_count: int = 0  # 缓存命中次数

    def fingerprint(self, tool_name: str, args: dict) -> str:
        """md5(tool_name + canonical_json(args))"""
        payload = tool_name + canonical_json(args)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def check(self, tool_name: str, args: dict) -> str | None:
        """命中返回缓存结果并 hit_count += 1；未命中返回 None。"""
        fp = self.fingerprint(tool_name, args)
        if fp not in self._records:
            return None
        self.hit_count += 1
        return self._records[fp]

    def save(self, tool_name: str, args: dict, result: str) -> None:
        """记录结果并 real_call_count += 1（仅在真实执行后调用）。"""
        fp = self.fingerprint(tool_name, args)
        self._records[fp] = result
        self.real_call_count += 1


class SemanticDeduper:
    """可选：语义级去重。默认实现为占位（normalized 字符串相似度），
    接口预留真实 embedding 实现。"""

    def is_duplicate(
        self,
        args: dict,
        history_args: list[dict],
        threshold: float = 0.95,
    ) -> bool:
        current = self._normalize(args)
        for past in history_args:
            if SequenceMatcher(None, current, self._normalize(past)).ratio() >= threshold:
                return True
        return False

    @staticmethod
    def _normalize(args: dict) -> str:
        text = canonical_json(args)
        text = re.sub(r"\s+", "", text)
        return text.lower()
