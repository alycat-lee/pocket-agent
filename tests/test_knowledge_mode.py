from __future__ import annotations

from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.intent_router import IntentRouter
from pocket_agent.knowledge import EMPTY_RESULT
from pocket_agent.runner import AgentRunner
from pocket_agent.session import SessionManager
from pocket_agent.task import TaskCard, infer_knowledge_mode
from pocket_agent.tools import ToolRegistry
from pocket_agent.verifier import verify


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))


def _mgr(mock_llm_factory: type, responses: list[object]) -> tuple[SessionManager, object, _Bus]:
    cfg = AgentConfig(
        max_steps=3,
        planner_enabled=False,
        verifier_enabled=True,
    )
    llm = mock_llm_factory(responses)
    bus = _Bus()
    mgr = SessionManager(
        cfg,
        llm,
        AgentRunner(cfg, llm, ToolRegistry(cfg)),
        events=bus,
        knowledge=None,
    )
    return mgr, llm, bus


def test_infer_knowledge_mode_three_tiers() -> None:
    assert infer_knowledge_mode("你是谁") == "none"
    assert infer_knowledge_mode("自我介绍") == "none"
    assert infer_knowledge_mode("根据文档介绍协议") == "strict"
    assert infer_knowledge_mode("知识库里有没有星核协议") == "strict"
    assert infer_knowledge_mode("今年市值多少") == "augment"
    assert infer_knowledge_mode("谁是作者") == "augment"


def test_who_are_you_asks_model_not_kb(mock_llm_factory: type) -> None:
    mgr, llm, _bus = _mgr(mock_llm_factory, ["我是 pocket-agent 默认助手", "身份"])
    answer = mgr.chat("s1", "你是谁")
    assert mgr.last_intent == "qa"
    assert mgr.last_task_card is not None
    assert mgr.last_task_card.knowledge_mode == "none"
    assert mgr.last_task_card.needs_knowledge is False
    assert answer == "我是 pocket-agent 默认助手"
    assert answer != EMPTY_RESULT
    assert llm.prompts


def test_strict_empty_kb_stays_honest(mock_llm_factory: type) -> None:
    mgr, _llm, _bus = _mgr(mock_llm_factory, ["标题"])
    answer = mgr.chat("s1", "根据文档介绍 XXX")
    assert mgr.last_task_card is not None
    assert mgr.last_task_card.knowledge_mode == "strict"
    assert answer == EMPTY_RESULT


def test_augment_empty_kb_falls_back_and_emits(
    mock_llm_factory: type,
) -> None:
    mgr, llm, bus = _mgr(mock_llm_factory, ["当前汇率大约七点二", "汇率"])
    answer = mgr.chat("s1", "今年最新的汇率是多少", force_mode="qa")
    assert mgr.last_task_card is not None
    assert mgr.last_task_card.knowledge_mode == "augment"
    assert answer == "当前汇率大约七点二"
    assert answer != EMPTY_RESULT
    assert any(name == "rag.empty" for name, _ in bus.events)
    empty = next(payload for name, payload in bus.events if name == "rag.empty")
    assert empty["query"]
    assert empty.get("turn") == 1
    assert verify(answer, mgr.last_task_card, evidence=EMPTY_RESULT) == []
    assert llm.prompts


def test_identity_rule_beats_short_chitchat() -> None:
    class _Llm:
        def generate(self, prompt: str) -> str:
            return "x"

        def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
            return 1

        def generate_json(self, prompt: str, schema: dict) -> dict:
            return {}

    mode, conf, source, reason = IntentRouter(_Llm()).classify("你是谁")
    assert mode == "qa"
    assert conf == 0.95
    assert source == "rule"
    assert reason == "身份问答"


def test_legacy_needs_knowledge_syncs_to_strict() -> None:
    card = TaskCard(type="qa", goal="旧卡", needs_knowledge=True)
    assert card.knowledge_mode == "strict"
    assert card.needs_knowledge is True
