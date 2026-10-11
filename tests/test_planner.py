from __future__ import annotations

from pocket_agent.config import AgentConfig
from pocket_agent.intent_router import IntentRouter
from pocket_agent.llm_client import LLMFormatError
from pocket_agent.runner import AgentRunner
from pocket_agent.session import SessionManager
from pocket_agent.task import TASK_CARD_SCHEMA, Planner, card_from_dict
from pocket_agent.tools import ToolRegistry


def test_planner_parses_card_and_normalizes_wanzi(mock_llm_factory: type) -> None:
    llm = mock_llm_factory(
        [
            {
                "type": "writer",
                "goal": "写一篇随意题材的长篇小说",
                "constraints": {"total_chars": "2万字", "language": "zh"},
                "plan": ["开篇", "发展", "高潮", "收束"],
                "needs_history": False,
            }
        ]
    )
    card = Planner(llm).plan("给我生成一篇2万字的小说 内容随意")
    assert card.type == "writer"
    assert card.constraints["total_chars"] == 20000
    assert card.needs_history is False
    assert "分析用户任务" in llm.prompts[0]
    assert "只输出 JSON" in llm.prompts[0]
    assert TASK_CARD_SCHEMA["required"] == [
        "type",
        "goal",
        "constraints",
        "plan",
        "needs_history",
    ]


def test_card_from_dict_fills_chars_from_user_input() -> None:
    card = card_from_dict(
        {
            "type": "writer",
            "goal": "写小说",
            "constraints": {},
            "plan": ["a", "b", "c"],
            "needs_history": False,
        },
        user_input="给我生成一篇2万字的小说 内容随意",
    )
    assert card.constraints["total_chars"] == 20000


def test_planner_failure_falls_back_to_rule_then_agent(
    mock_llm_factory: type,
) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=False)
    llm = mock_llm_factory(
        [LLMFormatError("bad card"), "天气晴朗可以出门散步。", "标题"]
    )
    runner = AgentRunner(cfg, llm, ToolRegistry(cfg))

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            del event, payload

    mgr = SessionManager(cfg, llm, runner, events=_Bus())
    mgr.chat("s1", "查北京天气")
    assert mgr.last_task_card is not None
    assert mgr.last_task_card.type == "agent"
    assert mgr.last_intent == "agent"

    llm2 = mock_llm_factory([LLMFormatError("bad card"), "你好呀今天怎么样。", "闲聊"])
    runner2 = AgentRunner(cfg, llm2, ToolRegistry(cfg))
    mgr2 = SessionManager(cfg, llm2, runner2, events=_Bus())
    mgr2.chat("s2", "你好")
    assert mgr2.last_task_card is not None
    assert mgr2.last_task_card.type == "qa"
    assert mgr2.last_intent == "qa"


def test_intent_router_cases_remain_as_fallback() -> None:
    """回归：规则路径仍可独立命中事故原句。"""
    router = IntentRouter(
        type(
            "L",
            (),
            {
                "generate": lambda self, p: "x",
                "generate_int": lambda self, p, a, b: 1,
                "generate_json": lambda self, p, s: {},
            },
        )()
    )
    mode, _c, source, _r = router.classify("给我生成一篇2万字的小说 内容随意")
    assert mode == "writer"
    assert source == "rule"
    assert router.try_rule(
        "请帮我分析一下这个数据集的质量问题，需要覆盖完整评估维度"
    ) is None
