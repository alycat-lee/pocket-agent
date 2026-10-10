from __future__ import annotations

import math

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMClientProtocol
from pocket_agent.prompts import SUMMARIZE

# 文档 02 §3.3 各固定区段字符预算
_SYSTEM_BUDGET = 300
_TASK_BUDGET = 200
_STATE_BUDGET = 500
_STEP_INFO_BUDGET = 400
_STATE_SUMMARY_LIMIT = 500


class ContextCompressor:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClientProtocol | None = None,
    ) -> None:
        self._config = config
        self._llm = llm

    def estimate_tokens(self, text: str) -> int:
        """粗估：len(text) / 1.5，向上取整。"""
        return math.ceil(len(text) / 1.5)

    def compress_result(self, result: str) -> str:
        """超过 per_result_limit 时规则截断，或可选走 LLM 摘要。"""
        limit = self._config.per_result_limit
        if len(result) <= limit:
            return result

        if self._config.use_llm_summarize and self._llm is not None:
            clipped = self._rule_truncate(result, self._config.summarize_input_limit)
            prompt = SUMMARIZE.format(limit=limit, text=clipped)
            summary = self._llm.generate(prompt)
            if len(summary) > limit:
                return self._rule_truncate(summary, limit)
            return summary

        return self._rule_truncate(result, limit)

    def build_prompt(
        self,
        *,
        system: str,
        task: str,
        state: str,
        step_info: str,
        history: list[str],
    ) -> str:
        """按文档 02 §3.3 的预算表拼装：固定区段先行，history 从最新往回塞。"""
        system_part = system[:_SYSTEM_BUDGET]
        task_part = task[:_TASK_BUDGET]
        state_part = state[:_STATE_BUDGET]
        step_part = step_info[:_STEP_INFO_BUDGET]

        fixed = f"{system_part}\n\n{task_part}\n\n{state_part}\n\n{step_part}"
        budget = self._config.prompt_char_budget
        sep = "\n\n"

        selected_newest_first: list[str] = []
        used = len(fixed)
        for item in reversed(history):
            cost = len(sep) + len(item)
            if used + cost > budget:
                break
            selected_newest_first.append(item)
            used += cost

        if not selected_newest_first:
            return fixed

        chronological = list(reversed(selected_newest_first))
        history_block = sep.join(chronological)
        return f"{fixed}{sep}{history_block}"

    def summarize_state(self, history: list[str], keep_last: int = 3) -> str:
        """生成"已完成 N 步：<最近 keep_last 步>"格式摘要。"""
        n = len(history)
        recent = history[-keep_last:] if keep_last > 0 else []
        body = "；".join(recent)
        text = f"已完成 {n} 步：{body}"
        if len(text) > _STATE_SUMMARY_LIMIT:
            return self._rule_truncate(text, _STATE_SUMMARY_LIMIT)
        return text

    @staticmethod
    def _rule_truncate(text: str, limit: int) -> str:
        """头部 70% + 截断标记 + 尾部 20%，总长 ≤ limit。"""
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text

        head_len = int(limit * 0.7)
        tail_len = int(limit * 0.2)

        while head_len + tail_len > 0:
            if head_len + tail_len >= len(text):
                return text[:limit]
            omitted = len(text) - head_len - tail_len
            marker = f"\n[...截断 {omitted} 字...]\n"
            candidate = text[:head_len] + marker + text[-tail_len:]
            if len(candidate) <= limit:
                return candidate

            overflow = len(candidate) - limit
            reduce_head = min(head_len, max(1, (overflow * 7) // 9))
            reduce_tail = min(tail_len, max(0, overflow - reduce_head))
            if reduce_head + reduce_tail == 0:
                break
            head_len -= reduce_head
            tail_len -= reduce_tail

        return text[:limit]
