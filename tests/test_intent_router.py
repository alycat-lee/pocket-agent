from __future__ import annotations

import json
from pathlib import Path

from pocket_agent.intent_router import IntentRouter
from pocket_agent.llm_client import LLMFormatError

_CASES_PATH = Path(__file__).parent / "fixtures" / "intent_cases.json"


class _ScriptedLlm:
    def __init__(self, idx: int) -> None:
        self._idx = idx

    def generate(self, prompt: str) -> str:
        return "x"

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return self._idx

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return {}


class _BoomLlm(_ScriptedLlm):
    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        raise LLMFormatError("bad")


def test_rule_writer_keywords():
    r = IntentRouter(_ScriptedLlm(0))
    mode, conf, source, reason = r.classify("写一篇关于新能源汽车的分析")
    assert mode == "writer"
    assert conf == 1.0
    assert source == "rule"
    assert reason == "写一篇"


def test_rule_qa_short_chitchat():
    r = IntentRouter(_ScriptedLlm(1))
    mode, conf, source, reason = r.classify("你好")
    assert mode == "qa"
    assert conf == 0.9
    assert source == "rule"
    assert reason == "短句闲聊"


def test_rule_tool_hint_prevents_qa():
    r = IntentRouter(_ScriptedLlm(0))
    # "查" is a tool hint → not qa → LLM fallback
    mode, conf, source, reason = r.classify("查北京天气")
    assert mode != "qa" or conf < 0.9
    assert source == "llm"
    assert reason == "LLM 分类"


def test_llm_fallback_agent():
    r = IntentRouter(_ScriptedLlm(1))
    mode, conf, source, reason = r.classify(
        "请帮我分析一下这个数据集的质量问题，需要详细说明"
    )
    assert mode == "agent"
    assert conf == 0.7
    assert source == "llm"
    assert reason == "LLM 分类"


def test_llm_fallback_qa():
    r = IntentRouter(_ScriptedLlm(0))
    mode, conf, source, reason = r.classify("请解释一下量子计算的基本原理")
    assert mode == "qa"
    assert conf == 0.7
    assert source == "llm"
    assert reason == "LLM 分类"


def test_llm_fallback_writer():
    r = IntentRouter(_ScriptedLlm(2))
    mode, conf, source, reason = r.classify(
        "请帮我准备一份关于新能源汽车的详细材料，需要覆盖技术路线和市场分析"
    )
    assert mode == "writer"
    assert conf == 0.7
    assert source == "llm"
    assert reason == "LLM 分类"


def test_fallback_on_llm_error():
    r = IntentRouter(_BoomLlm(0))
    mode, conf, source, reason = r.classify(
        "请帮我分析一下这个数据集的质量问题，需要详细说明"
    )
    assert mode == "agent"
    assert conf == 0.5
    assert source == "fallback"
    assert reason == "回退默认"


def test_fallback_empty_input():
    r = IntentRouter(_ScriptedLlm(1))
    mode, conf, source, reason = r.classify("   ")
    assert mode == "agent"
    assert conf == 0.5
    assert source == "fallback"
    assert reason == "空输入"


def test_accident_sentence_routes_to_writer():
    r = IntentRouter(_ScriptedLlm(1))
    mode, conf, source, reason = r.classify("给我生成一篇2万字的小说 内容随意")
    assert mode == "writer"
    assert conf == 1.0
    assert source == "rule"
    assert "小说" in reason


def test_intent_cases_regression():
    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    assert cases, "intent_cases.json 不能为空"
    assert cases[0]["input"] == "给我生成一篇2万字的小说 内容随意"
    router = IntentRouter(_ScriptedLlm(1))
    for case in cases:
        mode, _conf, source, _reason = router.classify(case["input"])
        assert mode == case["mode"], case
        assert source == case["source"], case
