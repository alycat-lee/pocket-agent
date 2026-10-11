from __future__ import annotations

from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.runner import AgentRunner
from pocket_agent.session import SessionManager
from pocket_agent.tools import ToolRegistry
from pocket_agent.verifier import verify

ACCIDENT = "给我生成一篇2万字的小说 内容随意"


def test_accident_sentence_full_chain(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, answer_max_retries=1)
    body = "林小满走进雨里。" + ("字" * 13000)
    llm = mock_llm_factory(
        [
            {
                "type": "writer",
                "goal": "生成一篇题材随意的两万字小说",
                "constraints": {"total_chars": 20000},
                "plan": ["开篇", "发展", "高潮", "收束"],
                "needs_history": False,
            },
            "小说标题",
        ]
    )
    runner = AgentRunner(cfg, llm, ToolRegistry(cfg))

    class _Writer:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def write(self, topic: str, **kwargs: Any) -> str:
            self.kwargs = {"topic": topic, **kwargs}
            return body

    writer = _Writer()
    planned: list[dict] = []

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            if event == "task.planned":
                planned.append(payload)

    mgr = SessionManager(cfg, llm, runner, events=_Bus(), writer=writer)
    answer = mgr.chat("s1", ACCIDENT)

    assert planned
    assert planned[0]["type"] == "writer"
    assert planned[0]["constraints"]["total_chars"] == 20000
    assert mgr.last_task_card is not None
    assert mgr.last_task_card.type == "writer"
    assert mgr.last_task_card.constraints["total_chars"] == 20000
    assert writer.kwargs.get("total_chars") == 20000
    assert writer.kwargs.get("plan") == ["开篇", "发展", "高潮", "收束"]
    assert "<think>" not in answer
    assert "嗯，" not in answer[:8]
    # 字数验收执行：短文会被打回；本用例正文超过 60% 目标才交付
    assert verify(answer, mgr.last_task_card) == []
    assert mgr.last_verify_reasons == []
    assert len(answer) >= int(20000 * 0.60)
