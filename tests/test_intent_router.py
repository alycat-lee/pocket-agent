from __future__ import annotations

from pocket_agent.intent_router import IntentRouter


class _ScriptedLlm:
    def __init__(self, idx: int) -> None:
        self._idx = idx

    def generate(self, prompt: str) -> str:
        return "x"

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return self._idx

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return {}


def test_rule_writer_keywords():
    r = IntentRouter(_ScriptedLlm(0))
    mode, conf = r.classify("写一篇关于新能源汽车的分析")
    assert mode == "writer"
    assert conf == 1.0


def test_rule_qa_short_chitchat():
    r = IntentRouter(_ScriptedLlm(1))
    mode, conf = r.classify("你好")
    assert mode == "qa"
    assert conf == 0.9


def test_rule_tool_hint_prevents_qa():
    r = IntentRouter(_ScriptedLlm(0))
    # "查" is a tool hint → not qa → LLM fallback
    mode, conf = r.classify("查北京天气")
    assert mode != "qa" or conf < 0.9


def test_llm_fallback_agent():
    r = IntentRouter(_ScriptedLlm(1))
    mode, conf = r.classify("请帮我分析一下这个数据集的质量问题，需要详细说明")
    assert mode == "agent"
    assert conf == 0.7


def test_llm_fallback_qa():
    r = IntentRouter(_ScriptedLlm(0))
    mode, conf = r.classify("请解释一下量子计算的基本原理")
    assert mode == "qa"
    assert conf == 0.7


def test_llm_fallback_writer():
    r = IntentRouter(_ScriptedLlm(2))
    mode, conf = r.classify("请帮我准备一份关于新能源汽车的详细材料，需要覆盖技术路线和市场分析")
    assert mode == "writer"
    assert conf == 0.7
