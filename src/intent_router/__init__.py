"""意图自动路由：规则快速路径 + LLM 兜底三分类。"""

from __future__ import annotations

import re
from typing import Literal

from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import INTENT_ROUTE

IntentMode = Literal["qa", "agent", "writer"]
IntentSource = Literal["rule", "llm", "fallback"]
ClassifyResult = tuple[IntentMode, float, IntentSource, str]

_WRITER_RE = re.compile(
    r"(?:写|生成|创作|来)(?:一|两|几)?[篇部段个]?.{0,8}(?:小说|故事|散文|诗歌|文章|报告|文案)"
)
# 正则未覆盖的遗留关键词（翻译/续写/「写一篇关于…分析」等）
_WRITER_KEYWORDS = (
    "写一篇",
    "写一份",
    "写一封",
    "生成文章",
    "撰写",
    "翻译",
    "续写",
    "扩写",
)
_QA_KEYWORDS = ("闲聊", "问候")
_QA_ENCYCLOPEDIA = re.compile(r"(?:什么是|是什么意思|请介绍一下|百科)")
_TOOL_HINTS = (
    "查询",
    "搜索",
    "获取",
    "调用",
    "读取",
    "写入",
    "写",
    "计算",
    "天气",
    "汇率",
    "查找",
    "检查",
    "执行",
    "删除",
    "发送",
    "创建",
    "更新",
    "查",
)
_MODES: tuple[IntentMode, ...] = ("qa", "agent", "writer")


class IntentRouter:
    """用户输入 → (mode, confidence, source, reason)。规则路径优先，LLM 兜底。"""

    def __init__(self, llm: LLMClientProtocol) -> None:
        self._llm = llm

    def classify(self, user_input: str) -> ClassifyResult:
        text = (user_input or "").strip()
        if not text:
            return "agent", 0.5, "fallback", "空输入"
        writer_match = _WRITER_RE.search(text)
        if writer_match is not None:
            return "writer", 1.0, "rule", writer_match.group(0)
        hit = next((kw for kw in _WRITER_KEYWORDS if kw in text), None)
        if hit is not None:
            return "writer", 1.0, "rule", hit
        has_tool = any(h in text for h in _TOOL_HINTS)
        if not has_tool:
            qa_hit = next((kw for kw in _QA_KEYWORDS if kw in text), None)
            if qa_hit is not None:
                return "qa", 0.9, "rule", qa_hit
            enc = _QA_ENCYCLOPEDIA.search(text)
            if enc is not None:
                return "qa", 0.9, "rule", enc.group(0)
        # 纯闲聊短句且无工具相关词 → qa
        if len(text) < 20 and not has_tool:
            return "qa", 0.9, "rule", "短句闲聊"
        # LLM 兜底三分类
        try:
            idx = self._llm.generate_int(
                INTENT_ROUTE.format(user_input=text),
                0,
                2,
            )
            if 0 <= idx <= 2:
                return _MODES[idx], 0.7, "llm", "LLM 分类"
        except LLMFormatError:
            pass
        except Exception:  # noqa: BLE001, S110
            pass
        return "agent", 0.5, "fallback", "回退默认"
