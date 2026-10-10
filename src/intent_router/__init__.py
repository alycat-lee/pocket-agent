"""意图自动路由：规则快速路径 + LLM 兜底三分类。"""

from __future__ import annotations

from typing import Literal

from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import INTENT_ROUTE

IntentMode = Literal["qa", "agent", "writer"]

_WRITER_KEYWORDS = (
    "写一篇",
    "写一份",
    "写一封",
    "生成文章",
    "撰写",
    "写一篇关于",
    "翻译",
    "续写",
    "扩写",
)
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
    """用户输入 → (mode, confidence)。规则路径优先，LLM 兜底。"""

    def __init__(self, llm: LLMClientProtocol) -> None:
        self._llm = llm

    def classify(self, user_input: str) -> tuple[IntentMode, float]:
        text = (user_input or "").strip()
        if not text:
            return "agent", 0.5
        # 规则快速路径：长文关键词 → writer
        if any(kw in text for kw in _WRITER_KEYWORDS):
            return "writer", 1.0
        # 纯闲聊短句且无工具相关词 → qa
        if len(text) < 20 and not any(h in text for h in _TOOL_HINTS):
            return "qa", 0.9
        # LLM 兜底三分类
        try:
            idx = self._llm.generate_int(
                INTENT_ROUTE.format(user_input=text),
                0,
                2,
            )
            if 0 <= idx <= 2:
                return _MODES[idx], 0.7
        except LLMFormatError:
            pass
        except Exception:  # noqa: BLE001, S110
            pass
        return "agent", 0.5
