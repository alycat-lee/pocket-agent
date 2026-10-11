"""多代理编排：TaskCard.plan → 依赖图 → worker 调度 → merger。"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pocket_agent.compressor import ContextCompressor
from pocket_agent.config import AgentConfig
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import LLMAbortedError, LLMClientProtocol, LLMFormatError
from pocket_agent.prompt_registry import variant_for_provider
from pocket_agent.prompts import SYNTHESIZE
from pocket_agent.runner import AgentRunner, EventEmitter, RunResult, StepRecord, StopReason
from pocket_agent.task import TaskCard
from pocket_agent.tools import ToolRegistry

MergeStrategy = Literal["collect", "chain", "synthesize", "vote"]
FailureAction = Literal["retry", "skip", "abort"]
NodeKind = Literal["worker", "merge"]
FailHook = Callable[["SubTask", int], None]

MAX_NESTING = 2
GLOBAL_STEP_BUDGET = 30
CLOUD_CONCURRENCY = 3
VOTE_PATHS = 3

_MERGE_RE = re.compile(r"(对比|比较|分析|总结|汇总|综合|合并|归纳|综述)")
_CHAIN_RE = re.compile(r"(基于|根据|然后|再|接着|随后|利用上述|用上面)")
_TOOLISH = ("查", "天气", "计算", "读取", "获取", "搜索", "调用", "查询")


class OrchestrationDepthError(RuntimeError):
    """嵌套深度超过上限：worker 不得再生 worker。"""


@dataclass
class SubTask:
    id: str
    index: int
    title: str
    kind: NodeKind
    depends_on: list[str] = field(default_factory=list)
    strategy: MergeStrategy | None = None


@dataclass
class WorkerResult:
    task_id: str
    answer: str
    steps: list[StepRecord] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False
    cache_hits: int = 0
    real_calls: int = 0
    total_steps: int = 0


@dataclass
class OrchestrationResult:
    answer: str
    steps: list[StepRecord]
    stop_reason: str
    stats: dict[str, Any]
    used: bool
    strategy: MergeStrategy
    workers: list[WorkerResult]
    graph: list[SubTask]


def has_product_flow(plan: list[str]) -> bool:
    """≥3 步且步骤间有产物传递（合并词 / 链式词 / 并列查询后汇总）。"""
    if len(plan) < 3:
        return False
    if any(_MERGE_RE.search(step) for step in plan[1:]):
        return True
    if any(_CHAIN_RE.search(step) for step in plan[1:]):
        return True
    heads = [_verb_head(step) for step in plan]
    return len(set(heads)) == 1 and bool(heads[0])


def should_orchestrate(card: TaskCard, *, depth: int = 0) -> bool:
    if depth >= MAX_NESTING:
        return False
    if card.type == "qa":
        return False
    if len(card.plan) < 3:
        return False
    if card.type == "writer" and not any(any(k in step for k in _TOOLISH) for step in card.plan):
        return False
    return has_product_flow(card.plan)


def build_graph(plan: list[str], *, strategy: MergeStrategy | None = None) -> list[SubTask]:
    """从计划推断依赖图：末步合并 / 链式依赖 / 并列 worker。"""
    if strategy == "vote":
        return [
            SubTask(id=f"v{i}", index=i, title=plan[0] if plan else "投票", kind="worker")
            for i in range(VOTE_PATHS)
        ]
    if len(plan) < 1:
        return []
    last_is_merge = len(plan) >= 3 and _MERGE_RE.search(plan[-1]) is not None
    if last_is_merge:
        workers = [
            SubTask(id=f"w{i}", index=i, title=title, kind="worker")
            for i, title in enumerate(plan[:-1])
        ]
        merge_strategy: MergeStrategy = (
            "synthesize" if _looks_like_write(plan[-1]) else "collect"
        )
        workers.append(
            SubTask(
                id=f"m{len(plan) - 1}",
                index=len(plan) - 1,
                title=plan[-1],
                kind="merge",
                depends_on=[n.id for n in workers],
                strategy=merge_strategy,
            )
        )
        return workers
    if any(_CHAIN_RE.search(step) for step in plan[1:]):
        nodes: list[SubTask] = []
        prev: str | None = None
        for i, title in enumerate(plan):
            nodes.append(
                SubTask(
                    id=f"w{i}",
                    index=i,
                    title=title,
                    kind="worker",
                    depends_on=[prev] if prev else [],
                )
            )
            prev = f"w{i}"
        return nodes
    return [
        SubTask(id=f"w{i}", index=i, title=title, kind="worker")
        for i, title in enumerate(plan)
    ]


def choose_strategy(card: TaskCard, graph: list[SubTask]) -> MergeStrategy:
    forced = card.constraints.get("merge") or card.constraints.get("strategy")
    if forced in {"collect", "chain", "synthesize", "vote"}:
        return forced  # type: ignore[return-value]
    if any(n.kind == "merge" and n.strategy == "synthesize" for n in graph):
        return "synthesize"
    if graph and all(n.depends_on for n in graph[1:]):
        return "chain"
    return "collect"


def merge_vote(answers: list[str]) -> str:
    """多数一致；平票取先出现的规范化答案。"""
    buckets: dict[str, list[str]] = defaultdict(list)
    for raw in answers:
        text = str(raw or "")
        key = " ".join(text.split())
        buckets[key].append(text)
    if not buckets:
        return ""
    winner = max(buckets.items(), key=lambda item: (len(item[1]), -list(buckets).index(item[0])))
    return winner[1][0]


def _verb_head(step: str) -> str:
    text = (step or "").strip()
    for token in _TOOLISH:
        if text.startswith(token) or token in text[:4]:
            return token
    return text[:2] if text else ""


def _looks_like_write(step: str) -> bool:
    return bool(re.search(r"(写|分析|对比|总结|综述)", step or ""))


class _ScopedBus:
    def __init__(self, inner: EventEmitter | None, extra: dict[str, Any]) -> None:
        self._inner = inner
        self._extra = extra

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._inner is None:
            return
        if event == "answer.done":
            return
        data = dict(payload)
        data.update(self._extra)
        self._inner.emit(event, data)


class Orchestrator:
    """编排器：独立 worker + 共享指纹 + 拓扑调度。"""

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClientProtocol,
        base_runner: AgentRunner,
        tools: ToolRegistry,
        *,
        events: EventEmitter | None = None,
        provider_type: str = "llamacpp",
        fail_hook: FailHook | None = None,
        step_budget: int = GLOBAL_STEP_BUDGET,
    ) -> None:
        self._config = config
        self._llm = llm
        self._base = base_runner
        self._tools = tools
        self._events = events
        self.provider_type = provider_type
        self._fail_hook = fail_hook
        self._step_budget = max(1, int(step_budget))
        self._compressor = ContextCompressor(config, llm)
        self._active: list[AgentRunner] = []
        self._remaining = self._step_budget
        self._run_turn: int | None = None

    def apply_config(self, config: AgentConfig) -> None:
        self._config = config
        self._compressor = ContextCompressor(config, self._llm)
        self._base.apply_config(config)

    def request_stop(self) -> None:
        self._base.request_stop()
        for runner in list(self._active):
            runner.request_stop()

    def should_orchestrate(self, card: TaskCard, *, depth: int = 0) -> bool:
        return should_orchestrate(card, depth=depth)

    def run(
        self,
        card: TaskCard,
        task: str,
        *,
        depth: int = 0,
        session_id: str = "default",
        permission: str = "workspace",
        strategy: MergeStrategy | None = None,
        turn: int | None = None,
    ) -> OrchestrationResult:
        prev_turn = self._run_turn
        self._run_turn = turn
        try:
            if depth >= MAX_NESTING:
                raise OrchestrationDepthError("worker 不得再生 worker")
            chosen = strategy or choose_strategy(
                card, build_graph(card.plan, strategy=strategy)
            )
            if not self.should_orchestrate(card, depth=depth) and chosen != "vote":
                return self._fast_path(
                    task, session_id=session_id, permission=permission
                )
            graph = build_graph(card.plan, strategy=chosen)
            if chosen == "vote":
                graph = build_graph(card.plan or [task], strategy="vote")
            return self._orchestrate(
                card,
                task,
                graph,
                chosen,
                depth=depth,
                session_id=session_id,
                permission=permission,
            )
        finally:
            self._run_turn = prev_turn

    def _fast_path(
        self,
        task: str,
        *,
        session_id: str,
        permission: str,
    ) -> OrchestrationResult:
        result = self._base.run(
            task,
            session_id=session_id,
            permission=permission,
            turn=self._run_turn,
        )
        return OrchestrationResult(
            answer=str(result.answer),
            steps=list(result.steps),
            stop_reason=result.stop_reason.value,
            stats=dict(result.stats),
            used=False,
            strategy="collect",
            workers=[],
            graph=[],
        )

    def _orchestrate(
        self,
        card: TaskCard,
        task: str,
        graph: list[SubTask],
        strategy: MergeStrategy,
        *,
        depth: int,
        session_id: str,
        permission: str,
    ) -> OrchestrationResult:
        self._remaining = self._step_budget
        shared_store: dict[str, str] = {}
        answers: dict[str, str] = {}
        results: dict[str, WorkerResult] = {}
        aborted = False
        self._emit(
            "orchestrator.start",
            {
                "session_id": session_id,
                "strategy": strategy,
                "depth": depth,
                "plan": list(card.plan),
                "workers": [
                    {"id": n.id, "title": n.title, "kind": n.kind} for n in graph
                ],
            },
        )
        pending = {n.id: n for n in graph}
        while pending and not aborted:
            ready = [
                n
                for n in pending.values()
                if all(dep in answers or _skipped(results.get(dep)) for dep in n.depends_on)
                and not any(_aborted_dep(results.get(dep)) for dep in n.depends_on)
            ]
            ready = [n for n in ready if not _blocked_by_failed_deps(n, results)]
            if not ready:
                break
            if self._remaining <= 0:
                for node in ready:
                    results[node.id] = WorkerResult(
                        task_id=node.id,
                        answer="",
                        error="全局步数预算耗尽",
                        skipped=True,
                    )
                    pending.pop(node.id, None)
                continue
            wave_workers = [n for n in ready if n.kind == "worker"]
            wave_merges = [n for n in ready if n.kind == "merge"]
            if wave_workers:
                produced = self._run_wave(
                    wave_workers,
                    card=card,
                    root_task=task,
                    answers=answers,
                    shared_store=shared_store,
                    session_id=session_id,
                    permission=permission,
                    depth=depth,
                )
                for node, wr in produced:
                    results[node.id] = wr
                    pending.pop(node.id, None)
                    if wr.error and not wr.skipped and wr.error == "abort":
                        aborted = True
                        break
                    if not wr.skipped and wr.error is None:
                        answers[node.id] = wr.answer
                        self._emit(
                            "task.step_done",
                            {"session_id": session_id, "index": node.index},
                        )
                if aborted:
                    break
            for node in wave_merges:
                wr = self._run_merge_node(
                    node,
                    card=card,
                    root_task=task,
                    answers=answers,
                    results=results,
                    session_id=session_id,
                )
                results[node.id] = wr
                pending.pop(node.id, None)
                if wr.error == "abort":
                    aborted = True
                    break
                if not wr.skipped and wr.error is None:
                    answers[node.id] = wr.answer
                    self._emit(
                        "task.step_done",
                        {"session_id": session_id, "index": node.index},
                    )

        ordered = [results[n.id] for n in graph if n.id in results]
        answer = self._final_answer(strategy, graph, answers, ordered, task)
        steps = [s for wr in ordered for s in wr.steps]
        real_calls = sum(wr.real_calls for wr in ordered)
        cache_hits = sum(wr.cache_hits for wr in ordered)
        stop_reason = "aborted" if aborted else "orchestrated"
        stats = {
            "total_steps": len(steps),
            "real_calls": real_calls,
            "cache_hits": cache_hits,
            "orchestrated": True,
            "strategy": strategy,
            "worker_count": sum(1 for n in graph if n.kind == "worker"),
        }
        self._emit(
            "orchestrator.done",
            {
                "session_id": session_id,
                "strategy": strategy,
                "worker_count": stats["worker_count"],
                "real_calls": real_calls,
                "cache_hits": cache_hits,
                "aborted": aborted,
            },
        )
        self._emit(
            "answer.done",
            {
                "answer": answer,
                "stop_reason": stop_reason,
                "total_steps": len(steps),
                "real_calls": real_calls,
                "cache_hits": cache_hits,
                "session_id": session_id,
            },
        )
        return OrchestrationResult(
            answer=answer,
            steps=steps,
            stop_reason=stop_reason,
            stats=stats,
            used=True,
            strategy=strategy,
            workers=ordered,
            graph=graph,
        )

    def _run_wave(
        self,
        nodes: list[SubTask],
        *,
        card: TaskCard,
        root_task: str,
        answers: dict[str, str],
        shared_store: dict[str, str],
        session_id: str,
        permission: str,
        depth: int,
    ) -> list[tuple[SubTask, WorkerResult]]:
        concurrency = self._concurrency()
        if concurrency <= 1 or len(nodes) == 1:
            return [
                (
                    node,
                    self._run_worker(
                        node,
                        card=card,
                        root_task=root_task,
                        answers=answers,
                        shared_store=shared_store,
                        session_id=session_id,
                        permission=permission,
                        depth=depth,
                    ),
                )
                for node in nodes
            ]
        out: list[tuple[SubTask, WorkerResult]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {
                pool.submit(
                    self._run_worker,
                    node,
                    card=card,
                    root_task=root_task,
                    answers=answers,
                    shared_store=shared_store,
                    session_id=session_id,
                    permission=permission,
                    depth=depth,
                ): node
                for node in nodes
            }
            for fut in as_completed(futs):
                node = futs[fut]
                out.append((node, fut.result()))
        order = {n.id: i for i, n in enumerate(nodes)}
        out.sort(key=lambda item: order[item[0].id])
        return out

    def _run_worker(
        self,
        node: SubTask,
        *,
        card: TaskCard,
        root_task: str,
        answers: dict[str, str],
        shared_store: dict[str, str],
        session_id: str,
        permission: str,
        depth: int,
    ) -> WorkerResult:
        del card, depth
        prompt = self._worker_prompt(node, answers, root_task)
        self._emit(
            "worker.start",
            {
                "session_id": session_id,
                "worker_id": node.id,
                "index": node.index,
                "task": prompt,
                "kind": node.kind,
                "title": node.title,
            },
        )
        last_error: str | None = None
        for attempt in range(2):
            try:
                if self._fail_hook is not None:
                    self._fail_hook(node, attempt)
                wr = self._execute_runner(
                    node,
                    prompt,
                    shared_store=shared_store,
                    session_id=session_id,
                    permission=permission,
                )
            except LLMAbortedError:
                wr = WorkerResult(task_id=node.id, answer="", error="abort")
            except Exception as exc:  # noqa: BLE001 — 子任务失败隔离
                wr = WorkerResult(task_id=node.id, answer="", error=str(exc))
            if wr.error is None:
                self._emit(
                    "worker.done",
                    {
                        "session_id": session_id,
                        "worker_id": node.id,
                        "answer": wr.answer,
                        "cache_hits": wr.cache_hits,
                        "real_calls": wr.real_calls,
                        "total_steps": wr.total_steps,
                    },
                )
                return wr
            last_error = wr.error
            if wr.error == "abort":
                return wr
        action = self._decide_failure(node, 1, last_error)
        if action == "abort":
            result = WorkerResult(task_id=node.id, answer="", error="abort")
        else:
            result = WorkerResult(
                task_id=node.id,
                answer="",
                error=last_error or "failed",
                skipped=True,
            )
        self._emit(
            "worker.done",
            {
                "session_id": session_id,
                "worker_id": node.id,
                "error": result.error,
                "skipped": result.skipped,
            },
        )
        return result

    def _execute_runner(
        self,
        node: SubTask,
        prompt: str,
        *,
        shared_store: dict[str, str],
        session_id: str,
        permission: str,
    ) -> WorkerResult:
        if self._remaining <= 0:
            return WorkerResult(
                task_id=node.id,
                answer="",
                error="全局步数预算耗尽",
                skipped=True,
            )
        max_steps = max(1, min(self._config.max_steps, self._remaining))
        cfg = replace(self._config, max_steps=max_steps)
        ledger = CallLedger(_records=shared_store)
        bus = _ScopedBus(self._events, {"worker_id": node.id, "session_id": session_id})
        runner = AgentRunner(cfg, self._llm, self._tools, events=bus)
        runner.approval_gate = self._base.approval_gate
        self._active.append(runner)
        try:
            result: RunResult = runner.run(
                prompt,
                ledger=ledger,
                session_id=session_id,
                permission=permission,
                turn=self._run_turn,
            )
        finally:
            if runner in self._active:
                self._active.remove(runner)
        used = int(result.stats.get("total_steps") or len(result.steps))
        self._remaining = max(0, self._remaining - used)
        error: str | None = None
        if result.stop_reason == StopReason.STOPPED:
            error = "abort"
        return WorkerResult(
            task_id=node.id,
            answer=str(result.answer),
            steps=list(result.steps),
            error=error,
            cache_hits=int(result.stats.get("cache_hits") or 0),
            real_calls=int(result.stats.get("real_calls") or 0),
            total_steps=used,
        )

    def _run_merge_node(
        self,
        node: SubTask,
        *,
        card: TaskCard,
        root_task: str,
        answers: dict[str, str],
        results: dict[str, WorkerResult],
        session_id: str,
    ) -> WorkerResult:
        strategy = node.strategy or choose_strategy(card, [node])
        parts = [
            answers[dep]
            for dep in node.depends_on
            if answers.get(dep)
        ]
        self._emit(
            "merger.start",
            {
                "session_id": session_id,
                "worker_id": node.id,
                "strategy": strategy,
                "inputs": list(node.depends_on),
                "title": node.title,
            },
        )
        last_error: str | None = None
        for attempt in range(2):
            try:
                if self._fail_hook is not None:
                    self._fail_hook(node, attempt)
                text = self._merge(strategy, parts, root_task or node.title)
                self._emit(
                    "merger.done",
                    {
                        "session_id": session_id,
                        "worker_id": node.id,
                        "strategy": strategy,
                        "answer": text,
                    },
                )
                return WorkerResult(task_id=node.id, answer=text)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        del results
        self._emit(
            "merger.done",
            {
                "session_id": session_id,
                "worker_id": node.id,
                "strategy": strategy,
                "error": last_error,
            },
        )
        return WorkerResult(task_id=node.id, answer="", error="abort")

    def _merge(self, strategy: MergeStrategy, parts: list[str], task: str) -> str:
        if strategy == "vote":
            return merge_vote(parts)
        if strategy == "chain":
            return parts[-1] if parts else ""
        if strategy == "synthesize":
            return self._synthesize(task, parts)
        return "\n\n".join(p for p in parts if p)

    def _synthesize(self, task: str, parts: list[str]) -> str:
        compressed = [self._compressor.compress_result(str(p)) for p in parts if p]
        summaries = "\n".join(f"- {item}" for item in compressed) or "（无）"
        prompt = SYNTHESIZE.format(task=task, summaries=summaries)
        try:
            return str(self._llm.generate(prompt)).strip()
        except LLMFormatError as exc:
            return f"[合并失败] {exc}"

    def _final_answer(
        self,
        strategy: MergeStrategy,
        graph: list[SubTask],
        answers: dict[str, str],
        ordered: list[WorkerResult],
        task: str,
    ) -> str:
        merge_nodes = [n for n in graph if n.kind == "merge"]
        if merge_nodes:
            last = merge_nodes[-1]
            if last.id in answers:
                return answers[last.id]
            failed = next((wr for wr in ordered if wr.task_id == last.id), None)
            if failed and failed.error == "abort":
                return "编排已终止：合并步骤失败。"
        worker_answers = [
            wr.answer for wr in ordered if not wr.skipped and wr.answer and wr.error is None
        ]
        if strategy == "vote":
            return merge_vote(worker_answers)
        if strategy == "chain":
            return worker_answers[-1] if worker_answers else ""
        if strategy == "synthesize":
            return self._synthesize(task, worker_answers)
        return "\n\n".join(worker_answers)

    def _worker_prompt(
        self,
        node: SubTask,
        answers: dict[str, str],
        root_task: str,
    ) -> str:
        """只注入本子任务 + 依赖节点的压缩摘要，不传其他 worker 全文。"""
        lines = [node.title]
        if node.depends_on:
            blobs: list[str] = []
            for dep in node.depends_on:
                raw = answers.get(dep)
                if not raw:
                    continue
                blobs.append(f"{dep}: {self._compressor.compress_result(str(raw))}")
            if blobs:
                lines.append("上游摘要：")
                lines.extend(blobs)
        elif root_task and root_task != node.title:
            goal = root_task[:200]
            lines.append(f"总目标：{goal}")
        return "\n".join(lines)

    def _decide_failure(
        self,
        node: SubTask,
        attempt: int,
        last_error: str | None,
    ) -> FailureAction:
        if last_error == "abort":
            return "abort"
        if attempt == 0:
            return "retry"
        if node.kind == "merge":
            return "abort"
        return "skip"

    def _concurrency(self) -> int:
        if variant_for_provider(self.provider_type) == "local":
            return 1
        return CLOUD_CONCURRENCY

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._events is not None:
            enriched = dict(payload)
            if self._run_turn is not None:
                enriched.setdefault("turn", self._run_turn)
            self._events.emit(event, enriched)


def _skipped(result: WorkerResult | None) -> bool:
    return result is not None and result.skipped


def _aborted_dep(result: WorkerResult | None) -> bool:
    return result is not None and result.error == "abort" and not result.skipped


def _blocked_by_failed_deps(node: SubTask, results: dict[str, WorkerResult]) -> bool:
    if node.kind != "merge":
        return False
    failed = [
        dep
        for dep in node.depends_on
        if dep in results and results[dep].error and not results[dep].skipped
    ]
    return bool(failed)
