from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from pocket_agent.checkpoint import CheckpointStore
from pocket_agent.compressor import ContextCompressor
from pocket_agent.config import AgentConfig
from pocket_agent.guard import LoopGuard
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import CHOOSE_TOOL, FILL_ARGS, FINAL_ANSWER, SYSTEM_BASE
from pocket_agent.tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)

# 写类工具审批：payload → True 批准 / False 拒绝（含超时）
ApprovalGate = Callable[[dict[str, Any]], bool]


class EventEmitter(Protocol):
    def emit(self, event: str, payload: dict[str, Any]) -> None: ...


class StopReason(Enum):
    MODEL_DONE = "model_done"  # choose 返回 -1
    MAX_STEPS = "max_steps"
    CONSECUTIVE_INTERCEPTS = "consecutive_intercepts"
    REPEAT_OUTPUT = "repeat_output"


@dataclass
class StepRecord:
    step: int
    tool_name: str
    args: dict
    result: str  # 压缩后的结果
    from_cache: bool
    error: str | None = None


@dataclass
class RunResult:
    answer: str
    steps: list[StepRecord]
    stop_reason: StopReason
    stats: dict  # total_steps / real_calls / cache_hits / intercepts / prompt_tokens_max


class AgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClientProtocol,
        registry: ToolRegistry,
        events: EventEmitter | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._registry = registry
        self._compressor = ContextCompressor(config, llm)
        self._ledger = CallLedger()
        self._guard = LoopGuard(config)
        self._checkpoints = CheckpointStore()
        self._events = events
        # 由 ui-web 等插件注入；None 时写工具无需审批（CLI/测试零回归）
        self.approval_gate: ApprovalGate | None = None

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._events is not None:
            self._events.emit(event, payload)

    def run(
        self,
        task: str,
        run_id: str | None = None,
        ledger: CallLedger | None = None,
    ) -> RunResult:
        """主循环：见规格 §8。仅 LLMConnectionError 允许上抛。

        run_id 非空时启用断点续跑：每步 dump；启动时恢复；正常完成后 clear。
        ledger：外部传入的当轮账本；为 None 时使用实例默认账本。
        """
        active_ledger = ledger if ledger is not None else self._ledger
        self._guard = LoopGuard(self._config)

        steps: list[StepRecord] = []
        start_step = 1
        prompt_tokens_max = 0
        intercepts = 0
        stop_reason = StopReason.MAX_STEPS

        if run_id:
            restored = self._restore_checkpoint(run_id, active_ledger)
            if restored is not None:
                steps, start_step, intercepts = restored

        for step in range(start_step, self._config.max_steps + 1):
            self._emit("step.start", {"step": step, "task": task})
            candidates = self._registry.recall(task, self._config.tool_recall_top_n)
            if not candidates:
                stop_reason = StopReason.MODEL_DONE
                break

            state = self._compressor.summarize_state(
                [self._format_step(s) for s in steps]
            )
            options = self._format_options(candidates)
            choose_prompt = self._build_choose_prompt(task, state, steps, options)
            prompt_tokens_max = max(
                prompt_tokens_max,
                self._compressor.estimate_tokens(choose_prompt),
            )

            t0 = time.perf_counter()
            try:
                choice = self._llm.generate_int(
                    choose_prompt,
                    minimum=-1,
                    maximum=len(candidates) - 1,
                )
            except LLMFormatError as exc:
                # 本步作废：记入历史，计 1 次拦截计入护栏，不向上抛
                result = f"[格式错误] {exc}"
                steps.append(
                    StepRecord(
                        step=step,
                        tool_name="(format_error)",
                        args={},
                        result=result,
                        from_cache=False,
                        error=result,
                    )
                )
                intercepts += 1
                self._emit(
                    "step.intercepted",
                    {"step": step, "reason": "format_error", "error": str(exc)},
                )
                if run_id:
                    self._dump_checkpoint(run_id, steps, active_ledger)
                logger.debug(
                    "step=%s tool=(format_error) args={{}} cache=miss "
                    "tokens≈%s elapsed=%.3fs",
                    step,
                    self._compressor.estimate_tokens(choose_prompt),
                    time.perf_counter() - t0,
                )
                verdict = self._guard.on_step(
                    step,
                    f"format_error:{exc}",
                    intercepted=True,
                )
                if verdict.should_stop:
                    stop_reason = StopReason(verdict.reason)
                    break
                continue

            if choice == -1:
                stop_reason = StopReason.MODEL_DONE
                break
            if choice < 0 or choice >= len(candidates):
                # 越界编号：同样按本步作废 + 拦截处理
                result = f"[格式错误] choice out of range: {choice}"
                steps.append(
                    StepRecord(
                        step=step,
                        tool_name="(format_error)",
                        args={"choice": choice},
                        result=result,
                        from_cache=False,
                        error=result,
                    )
                )
                intercepts += 1
                self._emit(
                    "step.intercepted",
                    {
                        "step": step,
                        "reason": "choice_out_of_range",
                        "choice": choice,
                    },
                )
                if run_id:
                    self._dump_checkpoint(run_id, steps, active_ledger)
                verdict = self._guard.on_step(
                    step,
                    f"format_error:choice={choice}",
                    intercepted=True,
                )
                if verdict.should_stop:
                    stop_reason = StopReason(verdict.reason)
                    break
                continue

            tool = candidates[choice]
            fill_prompt = self._build_fill_prompt(task, state, tool)
            prompt_tokens_max = max(
                prompt_tokens_max,
                self._compressor.estimate_tokens(fill_prompt),
            )

            try:
                args = self._llm.generate_json(fill_prompt, tool.schema)
            except LLMFormatError as exc:
                result = f"[工具错误] {exc}"
                record = StepRecord(
                    step=step,
                    tool_name=tool.name,
                    args={},
                    result=result,
                    from_cache=False,
                    error=result,
                )
                steps.append(record)
                if run_id:
                    self._dump_checkpoint(run_id, steps, active_ledger)
                logger.debug(
                    "step=%s tool=%s args=%s cache=miss tokens≈%s elapsed=%.3fs",
                    step,
                    tool.name,
                    {},
                    self._compressor.estimate_tokens(fill_prompt),
                    time.perf_counter() - t0,
                )
                verdict = self._guard.on_step(step, f"{choice}:{{}}", intercepted=False)
                if verdict.should_stop:
                    stop_reason = StopReason(verdict.reason)
                    break
                continue

            cached = active_ledger.check(tool.name, args)
            error: str | None = None
            if cached is not None:
                result = cached
                from_cache = True
                intercepts += 1
                self._emit(
                    "step.cache_hit",
                    {"step": step, "tool": tool.name, "args": args},
                )
                self._emit(
                    "step.intercepted",
                    {"step": step, "reason": "ledger_cache", "tool": tool.name},
                )
            else:
                from_cache = False
                if not tool.cacheable and self.approval_gate is not None:
                    approval_id = str(uuid.uuid4())
                    approval_payload = {
                        "approval_id": approval_id,
                        "step": step,
                        "tool": tool.name,
                        "args": args,
                        "timeout_sec": 60,
                    }
                    self._emit("approval.required", approval_payload)
                    if not self.approval_gate(approval_payload):
                        error = "[已拒绝] 用户未批准写操作"
                        result = error
                        intercepts += 1
                        self._emit(
                            "step.intercepted",
                            {
                                "step": step,
                                "reason": "approval_denied",
                                "tool": tool.name,
                                "approval_id": approval_id,
                            },
                        )
                        steps.append(
                            StepRecord(
                                step=step,
                                tool_name=tool.name,
                                args=args,
                                result=result,
                                from_cache=False,
                                error=error,
                            )
                        )
                        if run_id:
                            self._dump_checkpoint(run_id, steps, active_ledger)
                        verdict = self._guard.on_step(
                            step,
                            f"{choice}:{json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}",
                            intercepted=True,
                        )
                        if verdict.should_stop:
                            stop_reason = StopReason(verdict.reason)
                            break
                        continue
                try:
                    raw_result = tool.fn(**args)
                except Exception as exc:  # noqa: BLE001 — 规格要求捕获全部工具异常
                    error = f"[工具错误] {exc}"
                    raw_result = error
                result = self._compressor.compress_result(str(raw_result))
                active_ledger.save(tool.name, args, result)
                self._emit(
                    "step.tool_call",
                    {
                        "step": step,
                        "tool": tool.name,
                        "args": args,
                        "error": error,
                    },
                )

            steps.append(
                StepRecord(
                    step=step,
                    tool_name=tool.name,
                    args=args,
                    result=result,
                    from_cache=from_cache,
                    error=error,
                )
            )
            if run_id:
                self._dump_checkpoint(run_id, steps, active_ledger)

            logger.debug(
                "step=%s tool=%s args=%s cache=%s tokens≈%s elapsed=%.3fs",
                step,
                tool.name,
                args,
                "hit" if from_cache else "miss",
                self._compressor.estimate_tokens(choose_prompt),
                time.perf_counter() - t0,
            )

            raw_out = (
                f"{choice}:"
                f"{json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"
            )
            verdict = self._guard.on_step(step, raw_out, intercepted=from_cache)
            if verdict.should_stop:
                stop_reason = StopReason(verdict.reason)
                break
        else:
            stop_reason = StopReason.MAX_STEPS

        final_prompt = self._build_final_prompt(task, steps)
        prompt_tokens_max = max(
            prompt_tokens_max,
            self._compressor.estimate_tokens(final_prompt),
        )
        try:
            answer = self._llm.generate(final_prompt)
        except LLMFormatError as exc:
            answer = f"[生成回答失败] {exc}"

        if run_id:
            self._checkpoints.clear(run_id)

        self._emit(
            "answer.done",
            {
                "answer": answer,
                "stop_reason": stop_reason.value,
                "total_steps": len(steps),
            },
        )

        return RunResult(
            answer=answer,
            steps=steps,
            stop_reason=stop_reason,
            stats={
                "total_steps": len(steps),
                "real_calls": active_ledger.real_call_count,
                "cache_hits": active_ledger.hit_count,
                "intercepts": intercepts,
                "prompt_tokens_max": prompt_tokens_max,
            },
        )

    def _dump_checkpoint(
        self,
        run_id: str,
        steps: list[StepRecord],
        active_ledger: CallLedger,
    ) -> None:
        history = [
            {
                "step": s.step,
                "tool_name": s.tool_name,
                "args": s.args,
                "result": s.result,
                "from_cache": s.from_cache,
                "error": s.error,
            }
            for s in steps
        ]
        self._checkpoints.save(
            run_id,
            {
                "history": history,
                "ledger_records": active_ledger.export_records(),
                "step_index": steps[-1].step if steps else 0,
            },
        )

    def _restore_checkpoint(
        self,
        run_id: str,
        active_ledger: CallLedger,
    ) -> tuple[list[StepRecord], int, int] | None:
        """恢复 history 与 ledger；已执行步骤经账本校验标为缓存命中。

        返回 (steps, start_step, intercepts)；无有效 checkpoint 时返回 None。
        """
        ckpt = self._checkpoints.load(run_id)
        if ckpt is None:
            return None

        records = ckpt["ledger_records"]
        active_ledger.import_records(
            {str(k): str(v) for k, v in records.items()}
        )

        steps: list[StepRecord] = []
        intercepts = 0
        for item in ckpt["history"]:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", ""))
            args = item.get("args", {})
            if not isinstance(args, dict):
                args = {}
            result = str(item.get("result", ""))
            error = item.get("error")
            error_s = str(error) if error is not None else None
            step_no = int(item.get("step", len(steps) + 1))

            from_cache = bool(item.get("from_cache", False))
            if tool_name and tool_name != "(format_error)":
                cached = active_ledger.check(tool_name, args)
                if cached is not None:
                    result = cached
                    from_cache = True
                    intercepts += 1

            steps.append(
                StepRecord(
                    step=step_no,
                    tool_name=tool_name or "(format_error)",
                    args=args,
                    result=result,
                    from_cache=from_cache,
                    error=error_s,
                )
            )

        start_step = max(int(ckpt["step_index"]) + 1, 1)
        return steps, start_step, intercepts

    def _build_choose_prompt(
        self,
        task: str,
        state: str,
        steps: list[StepRecord],
        options: str,
    ) -> str:
        chrono = [self._format_step(s) for s in steps]
        selected = self._history_for_budget(
            chrono,
            system=SYSTEM_BASE,
            task=task,
            state=state,
            step_info=options,
        )
        history_newest_first = list(reversed(selected))
        history_text = "\n".join(history_newest_first) if history_newest_first else "（无）"
        body = CHOOSE_TOOL.format(
            task=task,
            state=state,
            history_text=history_text,
            options=options,
        )
        return f"{SYSTEM_BASE}\n\n{body}"

    def _build_fill_prompt(self, task: str, state: str, tool: Tool) -> str:
        schema_json = json.dumps(
            tool.schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        body = FILL_ARGS.format(
            task=task,
            state=state,
            tool_name=tool.name,
            schema_json=schema_json,
        )
        return f"{SYSTEM_BASE}\n\n{body}"

    def _build_final_prompt(self, task: str, steps: list[StepRecord]) -> str:
        chrono = [self._format_step(s) for s in steps]
        selected = self._history_for_budget(
            chrono,
            system=SYSTEM_BASE,
            task=task,
            state="",
            step_info="",
        )
        history_text = "\n".join(selected) if selected else "（无）"
        body = FINAL_ANSWER.format(task=task, history_text=history_text)
        return f"{SYSTEM_BASE}\n\n{body}"

    def _history_for_budget(
        self,
        chrono_lines: list[str],
        *,
        system: str,
        task: str,
        state: str,
        step_info: str,
    ) -> list[str]:
        packed = self._compressor.build_prompt(
            system=system,
            task=task,
            state=state,
            step_info=step_info,
            history=chrono_lines,
        )
        return [line for line in chrono_lines if line in packed]

    @staticmethod
    def _format_options(candidates: list[Tool]) -> str:
        return "\n".join(
            f"{i}. {tool.name}: {tool.short_desc}" for i, tool in enumerate(candidates)
        )

    @staticmethod
    def _format_step(record: StepRecord) -> str:
        cache_tag = " [缓存]" if record.from_cache else ""
        return (
            f"{record.step}. {record.tool_name}{cache_tag} "
            f"args={json.dumps(record.args, ensure_ascii=False, sort_keys=True)} "
            f"→ {record.result}"
        )
