from __future__ import annotations

import json
import logging
import threading
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
    STOPPED = "stopped"  # 用户中断


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
        self._stop = threading.Event()
        self._run_session_id: str = "default"

    def request_stop(self) -> None:
        """请求中断：下一步循环开头生效，随后 final_answer 优雅收尾。"""
        self._stop.set()

    def apply_config(self, config: AgentConfig) -> None:
        """热更新配置（settings / overrides 合并后即时生效）。"""
        self._config = config
        self._compressor = ContextCompressor(config, self._llm)
        self._guard = LoopGuard(config)
        if hasattr(self._llm, "_config"):
            self._llm._config = config
        if hasattr(self._registry, "_config"):
            self._registry._config = config

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._events is not None:
            enriched = dict(payload)
            enriched.setdefault("session_id", self._run_session_id)
            self._events.emit(event, enriched)

    def _step_fields(
        self,
        *,
        step: int,
        t0: float,
        prompt_tokens_est: int,
        current_tool: str | None = None,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "step_index": step,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "current_tool": current_tool,
            "prompt_tokens_est": prompt_tokens_est,
            "ttft_ms": None,  # 无 token 时间戳时为 null
        }

    def run(
        self,
        task: str,
        run_id: str | None = None,
        ledger: CallLedger | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """主循环：见规格 §8。仅 LLMConnectionError 允许上抛。

        run_id 非空时启用断点续跑：每步 dump；启动时恢复；正常完成后 clear。
        ledger：外部传入的当轮账本；为 None 时使用实例默认账本。
        session_id：事件落盘与轨迹视图关联的会话 id。
        """
        active_ledger = ledger if ledger is not None else self._ledger
        self._guard = LoopGuard(self._config)
        self._stop.clear()
        self._run_session_id = session_id or "default"

        steps: list[StepRecord] = []
        start_step = 1
        prompt_tokens_max = 0
        intercepts = 0
        stop_reason = StopReason.MAX_STEPS
        llm_ms = 0.0
        tool_ms = 0.0
        run_t0 = time.perf_counter()

        if run_id:
            restored = self._restore_checkpoint(run_id, active_ledger)
            if restored is not None:
                steps, start_step, intercepts = restored

        for step in range(start_step, self._config.max_steps + 1):
            if self._stop.is_set():
                stop_reason = StopReason.STOPPED
                break

            t0 = time.perf_counter()
            self._emit(
                "step.start",
                {
                    "task": task,
                    **self._step_fields(
                        step=step,
                        t0=t0,
                        prompt_tokens_est=0,
                        current_tool=None,
                    ),
                },
            )
            candidates = self._registry.recall(task, self._config.tool_recall_top_n)
            if not candidates:
                stop_reason = StopReason.MODEL_DONE
                break

            state = self._compressor.summarize_state(
                [self._format_step(s) for s in steps]
            )
            options = self._format_options(candidates)
            choose_prompt = self._build_choose_prompt(task, state, steps, options)
            choose_tokens = self._compressor.estimate_tokens(choose_prompt)
            prompt_tokens_max = max(prompt_tokens_max, choose_tokens)

            llm_t0 = time.perf_counter()
            try:
                choice = self._llm.generate_int(
                    choose_prompt,
                    minimum=-1,
                    maximum=len(candidates) - 1,
                )
            except LLMFormatError as exc:
                llm_ms += (time.perf_counter() - llm_t0) * 1000
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
                    {
                        "reason": "format_error",
                        "error": str(exc),
                        **self._step_fields(
                            step=step,
                            t0=t0,
                            prompt_tokens_est=choose_tokens,
                        ),
                    },
                )
                if run_id:
                    self._dump_checkpoint(run_id, steps, active_ledger)
                logger.debug(
                    "step=%s tool=(format_error) args={{}} cache=miss "
                    "tokens≈%s elapsed=%.3fs",
                    step,
                    choose_tokens,
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
            llm_ms += (time.perf_counter() - llm_t0) * 1000

            if choice == -1:
                stop_reason = StopReason.MODEL_DONE
                break
            if choice < 0 or choice >= len(candidates):
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
                        "reason": "choice_out_of_range",
                        "choice": choice,
                        **self._step_fields(
                            step=step,
                            t0=t0,
                            prompt_tokens_est=choose_tokens,
                        ),
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
            fill_tokens = self._compressor.estimate_tokens(fill_prompt)
            prompt_tokens_max = max(prompt_tokens_max, fill_tokens)

            llm_t0 = time.perf_counter()
            try:
                args = self._llm.generate_json(fill_prompt, tool.schema)
            except LLMFormatError as exc:
                llm_ms += (time.perf_counter() - llm_t0) * 1000
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
                    fill_tokens,
                    time.perf_counter() - t0,
                )
                verdict = self._guard.on_step(step, f"{choice}:{{}}", intercepted=False)
                if verdict.should_stop:
                    stop_reason = StopReason(verdict.reason)
                    break
                continue
            llm_ms += (time.perf_counter() - llm_t0) * 1000

            cached = active_ledger.check(tool.name, args)
            error: str | None = None
            if cached is not None:
                result = cached
                from_cache = True
                intercepts += 1
                meta = self._step_fields(
                    step=step,
                    t0=t0,
                    prompt_tokens_est=fill_tokens,
                    current_tool=tool.name,
                )
                self._emit(
                    "step.cache_hit",
                    {"tool": tool.name, "args": args, **meta},
                )
                self._emit(
                    "step.intercepted",
                    {"reason": "ledger_cache", "tool": tool.name, **meta},
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
                        "session_id": self._run_session_id,
                    }
                    self._emit("approval.required", approval_payload)
                    if not self.approval_gate(approval_payload):
                        error = "[已拒绝] 用户未批准写操作"
                        result = error
                        intercepts += 1
                        self._emit(
                            "step.intercepted",
                            {
                                "reason": "approval_denied",
                                "tool": tool.name,
                                "approval_id": approval_id,
                                **self._step_fields(
                                    step=step,
                                    t0=t0,
                                    prompt_tokens_est=fill_tokens,
                                    current_tool=tool.name,
                                ),
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
                tool_t0 = time.perf_counter()
                try:
                    raw_result = tool.fn(**args)
                except Exception as exc:  # noqa: BLE001 — 规格要求捕获全部工具异常
                    error = f"[工具错误] {exc}"
                    raw_result = error
                tool_ms += (time.perf_counter() - tool_t0) * 1000
                result = self._compressor.compress_result(str(raw_result))
                active_ledger.save(tool.name, args, result)
                self._emit(
                    "step.tool_call",
                    {
                        "tool": tool.name,
                        "args": args,
                        "error": error,
                        **self._step_fields(
                            step=step,
                            t0=t0,
                            prompt_tokens_est=fill_tokens,
                            current_tool=tool.name,
                        ),
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
                choose_tokens,
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

        if self._stop.is_set() and stop_reason != StopReason.STOPPED:
            stop_reason = StopReason.STOPPED

        final_prompt = self._build_final_prompt(task, steps)
        final_tokens = self._compressor.estimate_tokens(final_prompt)
        prompt_tokens_max = max(prompt_tokens_max, final_tokens)
        llm_t0 = time.perf_counter()
        try:
            answer = self._llm.generate(final_prompt)
        except LLMFormatError as exc:
            answer = f"[生成回答失败] {exc}"
        llm_ms += (time.perf_counter() - llm_t0) * 1000

        if run_id:
            self._checkpoints.clear(run_id)

        real_calls = active_ledger.real_call_count
        cache_hits = active_ledger.hit_count
        denom = real_calls + cache_hits
        if denom == 0:
            cache_hit_rate = 0.0
        else:
            cache_hit_rate = cache_hits / denom
            # 不谎报：仅真满命中才为 1.0
            if cache_hit_rate >= 1.0 and cache_hits != denom:
                cache_hit_rate = cache_hits / denom

        settlement = {
            "llm_ms": round(llm_ms, 3),
            "tool_ms": round(tool_ms, 3),
            "cache_hit_rate": cache_hit_rate,
            "real_calls": real_calls,
            "intercepts": intercepts,
            "prompt_tokens_max": prompt_tokens_max,
            "elapsed_ms": int((time.perf_counter() - run_t0) * 1000),
        }
        self._emit(
            "answer.done",
            {
                "answer": answer,
                "stop_reason": stop_reason.value,
                "total_steps": len(steps),
                **settlement,
            },
        )

        return RunResult(
            answer=answer,
            steps=steps,
            stop_reason=stop_reason,
            stats={
                "total_steps": len(steps),
                "real_calls": real_calls,
                "cache_hits": cache_hits,
                "intercepts": intercepts,
                "prompt_tokens_max": prompt_tokens_max,
                **settlement,
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
