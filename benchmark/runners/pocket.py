"""跑道 B：SessionManager 完整链路（Planner / 工具 / Writer / Verifier）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pocket_agent.config import AgentConfig

_TESTS = Path(__file__).resolve().parents[2] / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from pocket_agent.llm_client import LlamaCppClient, LLMConnectionError, LLMFormatError
from pocket_agent.runner import AgentRunner
from pocket_agent.session import SessionManager
from pocket_agent.tools import ToolRegistry
from pocket_agent.writer import LongWriter

from benchmark.runners import TrackResult, plan_only_fields
from benchmark.schema import task_turns
from fixtures.fixture_tools import register_fixture_tools


class _Bus:
    def emit(self, event: str, payload: dict[str, Any]) -> None:
        del event, payload


class PocketRunner:
    name = "pocket"

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self._config = AgentConfig(
            base_url=base_url,
            timeout=timeout,
            planner_enabled=True,
            verifier_enabled=True,
        )
        self._llm = LlamaCppClient(self._config)
        registry = ToolRegistry(self._config)
        register_fixture_tools(registry)
        self._agent = AgentRunner(self._config, self._llm, registry)
        self._writer = LongWriter(self._config, self._llm)

    def run_task(self, task: dict[str, Any]) -> TrackResult:
        tid = str(task["id"])
        dim = str(task["dimension"])
        bus = _Bus()
        mgr = SessionManager(
            self._config,
            self._llm,
            self._agent,
            events=bus,
            writer=self._writer,
        )
        session_id = f"bench-{tid}"
        answers: list[str] = []
        try:
            if plan_only_fields(task):
                card, _mode, _c, _src, _reason = mgr._resolve_card(
                    task_turns(task)[0], ""
                )
                mgr.last_task_card = card
                mgr.last_intent = card.type
                answer = card.goal
                extras = _extras(mgr)
                extras["plan_only"] = True
                return TrackResult(
                    track=self.name,
                    task_id=tid,
                    dimension=dim,
                    answer=answer,
                    turn_answers=[answer],
                    extras=extras,
                )
            for user in task_turns(task):
                answers.append(mgr.chat(session_id, user))
        except (LLMConnectionError, LLMFormatError) as exc:
            return TrackResult(
                track=self.name,
                task_id=tid,
                dimension=dim,
                answer=answers[-1] if answers else "",
                turn_answers=answers,
                extras=_extras(mgr),
                error=str(exc),
            )
        return TrackResult(
            track=self.name,
            task_id=tid,
            dimension=dim,
            answer=answers[-1] if answers else "",
            turn_answers=answers,
            extras=_extras(mgr),
        )


def _extras(mgr: SessionManager) -> dict[str, Any]:
    extras: dict[str, Any] = {
        "intent": mgr.last_intent,
        "verify_reasons": list(mgr.last_verify_reasons or []),
    }
    if mgr.last_task_card is not None:
        extras["task_card"] = mgr.last_task_card.to_dict()
    result = mgr.last_result
    if result is not None:
        real = int(result.stats.get("real_calls") or 0)
        extras["real_calls"] = real
        extras["steps"] = int(result.stats.get("total_steps") or 0)
        extras["intercepts"] = int(result.stats.get("intercepts") or 0)
        fps = {
            f"{s.tool_name}:{sorted(s.args.items())}"
            for s in result.steps
            if s.tool_name != "(format_error)"
        }
        extras["dup_call_rate"] = 0.0 if real == len(fps) else 1.0
        extras["tool_names"] = [s.tool_name for s in result.steps]
    return extras
