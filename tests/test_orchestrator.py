from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pocket_agent.config import AgentConfig
from pocket_agent.orchestrator import (
    OrchestrationDepthError,
    Orchestrator,
    SubTask,
    build_graph,
    choose_strategy,
    merge_vote,
    should_orchestrate,
)
from pocket_agent.plugins.orchestrator_plugin import OrchestratorPlugin
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context
from pocket_agent.session import SessionManager
from pocket_agent.task import TaskCard
from pocket_agent.tools import ToolRegistry, tool


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


def _card(plan: list[str], **kwargs: Any) -> TaskCard:
    data: dict[str, Any] = {
        "type": "agent",
        "goal": "测试任务",
        "constraints": {},
        "plan": plan,
        "needs_history": False,
    }
    data.update(kwargs)
    return TaskCard(
        type=data["type"],
        goal=str(data["goal"]),
        constraints=dict(data["constraints"]),
        plan=list(data["plan"]),
        needs_history=bool(data["needs_history"]),
    )


def _orch(
    llm: Any,
    *,
    events: _Bus | None = None,
    fail_hook: Any = None,
    provider_type: str = "llamacpp",
    registry: ToolRegistry | None = None,
) -> Orchestrator:
    cfg = AgentConfig(max_steps=5, verifier_enabled=False, planner_enabled=False)
    tools = registry if registry is not None else ToolRegistry(cfg)
    if registry is None and not tools.all():

        @tool(tools, "占位")
        def _noop(query: str = "") -> str:
            return query or "ok"

    base = AgentRunner(cfg, llm, tools)
    return Orchestrator(
        cfg,
        llm,
        base,
        tools,
        events=events,
        provider_type=provider_type,
        fail_hook=fail_hook,
    )


def test_fast_path_skips_when_plan_le_2(mock_llm_factory: type) -> None:
    llm = mock_llm_factory([-1, "单跑回答"])
    bus = _Bus()
    orch = _orch(llm, events=bus)
    card = _card(["查天气", "写结论"])
    assert should_orchestrate(card) is False
    result = orch.run(card, "查天气")
    assert result.used is False
    assert result.answer == "单跑回答"
    assert all(name != "orchestrator.start" for name, _ in bus.events)


def test_writer_novel_plan_does_not_orchestrate() -> None:
    card = _card(
        ["确定题材与人物", "展开开篇场景", "推进冲突", "推向高潮", "收束全文"],
        type="writer",
    )
    assert should_orchestrate(card) is False


def test_weather_graph_is_three_workers_plus_merge() -> None:
    plan = ["查北京天气", "查上海天气", "查广州天气", "写一段300字对比分析"]
    graph = build_graph(plan)
    assert [n.kind for n in graph] == ["worker", "worker", "worker", "merge"]
    assert graph[-1].depends_on == ["w0", "w1", "w2"]
    assert choose_strategy(_card(plan), graph) == "synthesize"


def test_topology_order_is_dependency_order(mock_llm_factory: type) -> None:
    llm = mock_llm_factory(
        [-1, "ALPHA_ONLY", -1, "BETA_ONLY", -1, "GAMMA_ONLY"]
    )
    bus = _Bus()
    orch = _orch(llm, events=bus)
    card = _card(["准备数据", "然后转换数据", "再输出结果"])
    result = orch.run(card, "流水线")
    starts = [p["title"] for e, p in bus.events if e == "worker.start"]
    assert starts == ["准备数据", "然后转换数据", "再输出结果"]
    beta_prompt = next(
        p["task"] for e, p in bus.events if e == "worker.start" and p["title"] == "然后转换数据"
    )
    gamma_prompt = next(
        p["task"] for e, p in bus.events if e == "worker.start" and p["title"] == "再输出结果"
    )
    assert "ALPHA_ONLY" in beta_prompt
    assert "ALPHA_ONLY" not in gamma_prompt
    assert "BETA_ONLY" in gamma_prompt
    assert result.used is True
    assert result.strategy == "chain"


def test_independent_workers_do_not_see_sibling_full_text(
    mock_llm_factory: type,
) -> None:
    llm = mock_llm_factory([-1, "UNIQUE_W0_FULL", -1, "UNIQUE_W1_FULL", "合并稿"])
    bus = _Bus()
    orch = _orch(llm, events=bus)
    card = _card(["查北京天气", "查上海天气", "写对比分析"])
    orch.run(card, "对比两地天气")
    w1 = next(p["task"] for e, p in bus.events if e == "worker.start" and p["title"] == "查上海天气")
    assert "UNIQUE_W0_FULL" not in w1


def test_shared_ledger_hits_across_workers(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=5, verifier_enabled=False)
    registry = ToolRegistry(cfg)
    calls: list[str] = []

    @tool(registry, "查询指定城市天气")
    def get_weather(city: str) -> str:
        calls.append(city)
        return f"天气:{city}"

    llm = mock_llm_factory(
        [
            0,
            {"city": "北京"},
            -1,
            "北京晴",
            0,
            {"city": "北京"},
            -1,
            "北京再确认",
        ]
    )
    bus = _Bus()
    orch = _orch(llm, events=bus, registry=registry)
    card = _card(["查北京天气", "再查北京天气确认", "汇总各地"])
    result = orch.run(card, "确认北京天气")
    assert calls == ["北京"]
    assert result.stats["cache_hits"] >= 1
    assert result.stats["real_calls"] == 1


def test_failure_retry_then_success(mock_llm_factory: type) -> None:
    llm = mock_llm_factory([-1, "ok-a", -1, "ok-b", -1, "ok-c"])
    seen: list[tuple[str, int]] = []

    def hook(node: SubTask, attempt: int) -> None:
        seen.append((node.id, attempt))
        if node.id == "w0" and attempt == 0:
            raise RuntimeError("transient")

    orch = _orch(llm, fail_hook=hook)
    card = _card(["准备数据", "然后转换数据", "再输出结果"])
    result = orch.run(card, "链")
    assert ("w0", 0) in seen
    assert ("w0", 1) in seen
    assert result.used is True
    assert result.answer == "ok-c"
    assert all(not wr.skipped for wr in result.workers)


def test_failure_skip_keeps_siblings(mock_llm_factory: type) -> None:
    llm = mock_llm_factory([-1, "北京晴", -1, "广州雨", "对比稿"])

    def hook(node: SubTask, attempt: int) -> None:
        del attempt
        if node.id == "w1":
            raise RuntimeError("shanghai down")

    bus = _Bus()
    orch = _orch(llm, events=bus, fail_hook=hook)
    card = _card(["查北京天气", "查上海天气", "查广州天气", "写对比分析"])
    result = orch.run(card, "三地天气")
    skipped = [wr for wr in result.workers if wr.skipped]
    assert len(skipped) == 1
    assert skipped[0].task_id == "w1"
    assert result.stop_reason != "aborted"
    assert "对比稿" in result.answer or result.answer


def test_failure_abort_on_merge(mock_llm_factory: type) -> None:
    llm = mock_llm_factory([-1, "a", -1, "b", -1, "c"])

    def hook(node: SubTask, attempt: int) -> None:
        del attempt
        if node.kind == "merge":
            raise RuntimeError("merge boom")

    orch = _orch(llm, fail_hook=hook)
    card = _card(["查北京天气", "查上海天气", "查广州天气", "写对比分析"])
    result = orch.run(card, "三地")
    assert result.stop_reason == "aborted"
    assert "终止" in result.answer


def test_depth_limit_blocks_nested_workers(mock_llm_factory: type) -> None:
    llm = mock_llm_factory([])
    orch = _orch(llm)
    card = _card(["查北京天气", "查上海天气", "写对比分析"])
    with pytest.raises(OrchestrationDepthError):
        orch.run(card, "嵌套", depth=2)
    assert should_orchestrate(card, depth=2) is False


def test_vote_majority_wins(mock_llm_factory: type) -> None:
    llm = mock_llm_factory([-1, "答案是42", -1, "答案是7", -1, "答案是42"])
    orch = _orch(llm)
    card = _card(["推理这道题"], constraints={"merge": "vote"})
    result = orch.run(card, "1+1", strategy="vote")
    assert result.used is True
    assert result.strategy == "vote"
    assert merge_vote(["答案是42", "答案是7", "答案是42"]) == "答案是42"
    assert result.answer == "答案是42"
    assert len(result.workers) == 3


def test_session_two_step_stays_on_single_runner(mock_llm_factory: type) -> None:
    cfg = AgentConfig(planner_enabled=False, verifier_enabled=False, max_steps=3)
    llm = mock_llm_factory([-1, "单runner"])
    tools = ToolRegistry(cfg)
    runner = AgentRunner(cfg, llm, tools)
    bus = _Bus()
    orch = Orchestrator(cfg, llm, runner, tools, events=bus)
    mgr = SessionManager(cfg, llm, runner, events=None, orchestrator=orch)
    mgr.chat("s1", "现在几点了")
    assert all(name != "orchestrator.start" for name, _ in bus.events)


def test_plugin_unmount_removes_orchestrator(mock_llm_factory: type) -> None:
    class _Llm:
        name = "llm-fake"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["llm"]

        def apply(self, ctx: Context, config: dict) -> None:
            del config
            ctx.provide("llm", mock_llm_factory([-1, "x"]))

    class _Agent:
        name = "agent-loop"
        requires: ClassVar[list[str]] = ["llm"]
        provides: ClassVar[list[str]] = ["agent"]

        def apply(self, ctx: Context, config: dict) -> None:
            cfg = AgentConfig()
            llm = ctx.require("llm")
            runner = AgentRunner(cfg, llm, ToolRegistry(cfg))
            ctx.provide("agent", runner)

    ctx = Context()
    ctx.mount(_Llm())
    ctx.mount(_Agent())
    ctx.mount(OrchestratorPlugin())
    assert ctx.has("orchestrator")
    ctx.unmount("orchestrator")
    assert not ctx.has("orchestrator")
    ctx.unmount_all()
