from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pocket_agent.config import AgentConfig
from pocket_agent.intent_router import IntentMode, IntentRouter, has_tool_hint
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import LLMAbortedError, LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import (
    FINAL_ANSWER,
    REWRITE_QUERY,
    ROLLING_SUMMARY,
    SESSION_TITLE,
)
from pocket_agent.runner import AgentRunner, EventEmitter, StepRecord
from pocket_agent.task import DEFAULT_WRITER_PLAN, Planner, TaskCard, default_card
from pocket_agent.verifier import attach_degrade, verify
from pocket_agent.workspace import WorkspaceManager

_TURN_ANSWER_DIGEST = 120

_SID_SAFE = re.compile(r"[^\w.\-]+")


def _safe_sid(session_id: str) -> str:
    return _SID_SAFE.sub("_", session_id.strip()) or "default"


def _digest_answer(answer: str, limit: int = _TURN_ANSWER_DIGEST) -> str:
    """轮次摘要：只保留回答短摘，去掉工具步骤式长文。"""
    text = " ".join((answer or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class WriterProtocol(Protocol):
    def write(self, topic: str, **kwargs: Any) -> str: ...


@dataclass
class TurnRecord:
    turn: int
    query_raw: str  # 用户原始输入
    query_rewritten: str  # 指代补全后的完整问题
    answer: str


@dataclass
class SessionState:
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)  # 最近 keep_full_turns 轮
    rolling_summary: str = ""  # 更早轮次的滚动摘要
    write_op_ledger: CallLedger = field(default_factory=CallLedger)
    title: str = "新会话"
    _last_permission: str = "workspace"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionManager:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClientProtocol,
        runner: AgentRunner,
        events: EventEmitter | None = None,
        *,
        intent_router: IntentRouter | None = None,
        writer: WriterProtocol | None = None,
        workspace: WorkspaceManager | None = None,
        sessions_dir: Path | str | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._runner = runner
        self._events = events
        self._intent_router = intent_router or IntentRouter(llm)
        self._planner = Planner(llm)
        self._writer = writer
        self._workspace = workspace
        self._sessions_dir = Path(sessions_dir) if sessions_dir else None
        self._sessions: dict[str, SessionState] = {}
        if self._sessions_dir is not None:
            self._load_all()
        # 测试可观测字段
        self.last_rewritten: str = ""
        self.last_run_task: str = ""
        self.last_result = None
        self.last_intent: IntentMode | None = None
        self.last_task_card: TaskCard | None = None
        self.last_verify_reasons: list[str] = []
        self.rewrite_call_count: int = 0
        self.summary_call_count: int = 0

    def get_state(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> list[dict]:
        """供 Web UI 列举会话，按 updated_at 降序。"""
        rows = [
            {
                "session_id": state.session_id,
                "title": state.title,
                "turns": len(state.turns),
                "rolling_summary": state.rolling_summary,
                "workspace": (
                    self._workspace.get_session_workspace(state.session_id).name
                    if self._workspace is not None
                    else "default"
                ),
                "permission": state._last_permission,
                "updated_at": state.updated_at,
            }
            for state in self._sessions.values()
        ]
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        return rows

    def delete_session(self, session_id: str) -> bool:
        """删除会话（内存 + 落盘文件）。"""
        self._sessions.pop(session_id, None)
        if self._sessions_dir is not None:
            path = self._sessions_dir / f"{_safe_sid(session_id)}.json"
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def _save_session(self, state: SessionState) -> None:
        """原子写入会话到 .sessions/{session_id}.json。"""
        if self._sessions_dir is None:
            return
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": state.session_id,
            "title": state.title,
            "turns": [
                {
                    "turn": t.turn,
                    "query_raw": t.query_raw,
                    "query_rewritten": t.query_rewritten,
                    "answer": t.answer,
                }
                for t in state.turns
            ],
            "rolling_summary": state.rolling_summary,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }
        path = self._sessions_dir / f"{_safe_sid(state.session_id)}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_all(self) -> None:
        """启动时扫描 .sessions/ 加载历史会话；损坏文件跳过。"""
        if not self._sessions_dir.is_dir():
            return
        for path in self._sessions_dir.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or "session_id" not in data:
                continue
            sid = str(data["session_id"])
            turns = [
                TurnRecord(
                    turn=int(t.get("turn", 0)),
                    query_raw=str(t.get("query_raw", "")),
                    query_rewritten=str(t.get("query_rewritten", "")),
                    answer=str(t.get("answer", "")),
                )
                for t in data.get("turns", [])
                if isinstance(t, dict)
            ]
            state = SessionState(
                session_id=sid,
                turns=turns,
                rolling_summary=str(data.get("rolling_summary", "")),
                title=str(data.get("title", "新会话")),
                created_at=float(data.get("created_at", time.time())),
                updated_at=float(data.get("updated_at", time.time())),
            )
            self._sessions[sid] = state

    def chat(
        self,
        session_id: str,
        user_input: str,
        *,
        force_mode: IntentMode | None = None,
        permission: str = "workspace",
        workspace_id: str | None = None,
    ) -> str:
        """多轮对话入口。见文档 07 §1。

        force_mode：用户纠正意图时传入，跳过自动分类。
        permission：readonly / workspace / full，映射审批策略。
        workspace_id：绑定工作区（None 则不变）。
        """
        state = self.get_state(session_id)
        rewritten: str | None = None
        source = "fallback"
        reason = "无事件总线"
        if self._events is None:
            mode: IntentMode = "agent"
            confidence = 1.0
            card = default_card("agent", user_input, user_input=user_input)
        elif force_mode is not None:
            mode = force_mode
            confidence = 1.0
            source = "manual"
            reason = "用户纠正"
            last = state.turns[-1] if state.turns else None
            if last is not None and last.query_raw == user_input:
                rewritten = last.query_rewritten
                state.turns.pop()
            card = default_card(mode, rewritten or user_input, user_input=user_input)
        else:
            card, mode, confidence, source, reason = self._resolve_card(
                user_input, state.rolling_summary
            )
        self.last_intent = mode
        self.last_task_card = card
        turn_no = len(state.turns) + 1
        if self._events is not None:
            self._events.emit(
                "task.planned",
                {
                    "session_id": session_id,
                    "turn": turn_no,
                    "query": user_input,
                    **card.to_dict(),
                    "plan_done": [False] * len(card.plan),
                },
            )
            self._events.emit(
                "intent.detected",
                {
                    "session_id": session_id,
                    "mode": mode,
                    "confidence": confidence,
                    "source": source,
                    "reason": reason,
                    "turn": turn_no,
                    "query": user_input,
                },
            )

        # 工作区绑定
        if workspace_id is not None and self._workspace is not None:
            self._workspace.bind_session(session_id, workspace_id)
            if self._events is not None:
                self._events.emit(
                    "session.workspace_changed",
                    {"session_id": session_id, "workspace_id": workspace_id},
                )
        # 权限变更事件
        if self._events is not None and permission != state._last_permission:
            state._last_permission = permission
            self._events.emit(
                "session.permission_changed",
                {"session_id": session_id, "permission": permission},
            )

        if rewritten is None:
            if card.needs_history and card.type == "agent":
                rewritten = self._rewrite_if_needed(state, user_input)
            elif card.needs_history and card.type == "qa":
                rewritten = self._rewrite_if_needed(
                    state, user_input, summary_only=True
                )
            else:
                rewritten = user_input
        self.last_rewritten = rewritten

        run_task = self._compose_run_task(state, rewritten, card=card)
        self.last_run_task = run_task

        answer = self._dispatch(card, run_task, session_id, permission, state)
        answer = self._verify_answer(answer, card, run_task, session_id)

        turn = TurnRecord(
            turn=turn_no,
            query_raw=user_input,
            query_rewritten=rewritten,
            answer=answer,
        )
        state.turns.append(turn)
        state.updated_at = time.time()
        self._maybe_rename(state, user_input)
        self._maybe_roll_summary(state)
        self._save_session(state)
        return answer

    def _resolve_card(
        self,
        user_input: str,
        rolling_summary: str,
    ) -> tuple[TaskCard, IntentMode, float, str, str]:
        if self._config.planner_enabled:
            try:
                card = self._ground_card(
                    self._planner.plan(user_input, rolling_summary),
                    user_input,
                )
                return card, card.type, 0.85, "planner", "任务卡"
            except LLMFormatError:
                pass
            except Exception:  # noqa: BLE001, S110 — Planner 失败走规则兜底
                pass
            ruled = self._intent_router.try_rule(user_input)
            if ruled is not None:
                mode, confidence, source, reason = ruled
                return (
                    default_card(mode, user_input, user_input=user_input),
                    mode,
                    confidence,
                    source,
                    reason,
                )
            return (
                default_card("agent", user_input, user_input=user_input),
                "agent",
                0.5,
                "fallback",
                "回退默认",
            )
        mode, confidence, source, reason = self._intent_router.classify(user_input)
        return (
            default_card(mode, user_input, user_input=user_input),
            mode,
            confidence,
            source,
            reason,
        )

    def _ground_card(self, card: TaskCard, user_input: str) -> TaskCard:
        """用原文硬线索校正小模型任务卡：写长文 / 工具调用不得落成闲聊。"""
        ruled = self._intent_router.try_rule(user_input)
        new_type = card.type
        if ruled is not None and ruled[0] == "writer":
            new_type = "writer"
        elif card.type == "qa" and has_tool_hint(user_input):
            new_type = "agent"
        plan = list(card.plan)
        if new_type == "writer" and not plan:
            plan = list(DEFAULT_WRITER_PLAN)
        if new_type == card.type and plan == list(card.plan):
            return card
        return TaskCard(
            type=new_type,
            goal=card.goal,
            constraints=card.constraints,
            plan=plan,
            needs_history=card.needs_history or new_type == "agent",
        )

    def _dispatch(
        self,
        card: TaskCard,
        run_task: str,
        session_id: str,
        permission: str,
        state: SessionState,
    ) -> str:
        if card.type == "writer" and self._writer is not None:
            return self._run_writer(card, run_task, session_id)
        if card.type == "qa":
            try:
                answer = str(self._llm.generate(run_task)).strip()
            except LLMAbortedError:
                answer = "（已停止）"
                self.last_result = None
                self._emit_answer_done(answer, stop_reason="stopped")
                return answer
            except LLMFormatError as exc:
                answer = f"[生成回答失败] {exc}"
            self.last_result = None
            self._emit_answer_done(answer, stop_reason="qa")
            return answer
        turn_ledger = CallLedger()
        turn_ledger.merge_records(state.write_op_ledger)
        result = self._runner.run(
            self._agent_task(card, run_task),
            ledger=turn_ledger,
            session_id=session_id,
            permission=permission,
        )
        self.last_result = result
        self._sync_write_ops(state, result.steps)
        self._mark_plan_progress(session_id, len(result.steps))
        return str(result.answer)

    def _run_writer(self, card: TaskCard, topic: str, session_id: str) -> str:
        writer = self._writer
        if writer is None:
            return topic
        if self._events is not None and hasattr(writer, "on_thinking"):
            def _emit_think(text: str) -> None:
                if text and self._config.reasoning_enabled and self._events is not None:
                    self._events.emit(
                        "step.thinking",
                        {"text": text, "session_id": session_id},
                    )

            writer.on_thinking = _emit_think
        if self._events is not None and hasattr(writer, "on_section"):
            def _on_section(index: int, _section: object, _body: str) -> None:
                if self._events is not None:
                    self._events.emit(
                        "task.step_done",
                        {
                            "session_id": session_id,
                            "index": max(0, index - 1),
                        },
                    )

            writer.on_section = _on_section
        raw = card.constraints.get("total_chars")
        total_chars = (
            int(raw)
            if isinstance(raw, (int, float)) and int(raw) > 0
            else max(1, self._config.section_char_target * 6)
        )
        goal = card.goal or topic
        try:
            answer = writer.write(
                goal,
                total_chars=total_chars,
                plan=list(card.plan) or None,
            )
        except LLMAbortedError:
            answer = "\n\n".join(writer.last_sections)
            writer.stopped = True
        self.last_result = None
        reason = "stopped" if getattr(writer, "stopped", False) else "writer"
        self._emit_answer_done(answer, stop_reason=reason)
        return answer

    def _agent_task(self, card: TaskCard, run_task: str) -> str:
        extras = [
            f"{k}={v}"
            for k, v in card.constraints.items()
            if v is not None and str(v).strip()
        ]
        if not extras:
            return run_task
        return f"{run_task}\n约束：{'；'.join(extras)}"

    def _verify_answer(
        self,
        answer: str,
        card: TaskCard,
        run_task: str,
        session_id: str,
    ) -> str:
        self.last_verify_reasons = []
        if not self._config.verifier_enabled:
            return answer
        reasons = verify(answer, card)
        retries = max(0, int(self._config.answer_max_retries))
        if card.type == "writer":
            retries = 0
        attempt = 0
        while reasons and attempt < retries:
            self.last_verify_reasons = list(reasons)
            self._emit_rejected(session_id, reasons)
            prompt = FINAL_ANSWER.format(task=run_task, history_text="（无）")
            prompt = (
                f"{prompt}\n\n验收未通过：{'；'.join(reasons)}\n"
                "请修正后重新输出回答。"
            )
            try:
                answer = str(self._llm.generate(prompt)).strip() or answer
            except LLMAbortedError:
                break
            except LLMFormatError:
                break
            reasons = verify(answer, card)
            attempt += 1
        if reasons:
            self.last_verify_reasons = list(reasons)
            self._emit_rejected(session_id, reasons)
            return attach_degrade(answer, reasons)
        return answer

    def _emit_rejected(self, session_id: str, reasons: list[str]) -> None:
        if self._events is not None:
            self._events.emit(
                "answer.rejected",
                {"session_id": session_id, "reasons": reasons},
            )

    def _emit_answer_done(self, answer: str, *, stop_reason: str) -> None:
        if self._events is None:
            return
        meta = getattr(self, "_run_meta", None)
        extra = dict(meta) if isinstance(meta, dict) else {}
        self._events.emit(
            "answer.done",
            {
                "answer": answer,
                "stop_reason": stop_reason,
                "total_steps": 0,
                "llm_ms": 0,
                "tool_ms": 0,
                "cache_hit_rate": 0.0,
                "real_calls": 0,
                "intercepts": 0,
                "prompt_tokens_max": 0,
                "elapsed_ms": 0,
                **extra,
            },
        )

    def _mark_plan_progress(self, session_id: str, steps: int) -> None:
        card = self.last_task_card
        if self._events is None or card is None or not card.plan:
            return
        for index in range(min(steps, len(card.plan))):
            self._events.emit(
                "task.step_done",
                {"session_id": session_id, "index": index},
            )

    def _maybe_rename(self, state: SessionState, user_input: str) -> None:
        """首轮问答完成后自动生成会话标题；失败回退为用户输入前 12 字截断。

        仅在接入事件总线时触发——单元测试无 events 时不消耗测试替身队列。
        """
        if self._events is None:
            return
        # 仅在首轮后、标题仍为默认值时触发
        if len(state.turns) != 1 or state.title != "新会话":
            return
        title = ""
        try:
            prompt = SESSION_TITLE.format(user_input=user_input)
            raw = str(self._llm.generate(prompt)).strip()
            # 取首行、去引号、限 12 字
            title = raw.splitlines()[0].strip().strip("\"'""") if raw else ""
            if title:
                title = title[:12]
        except Exception:  # noqa: BLE001 — 标题生成永不阻塞主流程
            title = ""
        if not title:
            title = user_input[:12]
        state.title = title
        if self._events is not None:
            self._events.emit(
                "session.renamed",
                {"session_id": state.session_id, "title": title},
            )

    def _rewrite_if_needed(
        self,
        state: SessionState,
        user_input: str,
        *,
        summary_only: bool = False,
    ) -> str:
        if not state.turns and not state.rolling_summary:
            return user_input
        if summary_only:
            summary = state.rolling_summary.strip()
            if not summary:
                return user_input
            history_text = f"摘要：{summary}"
        else:
            history_text = self._history_text(state)
        prompt = REWRITE_QUERY.format(
            history_text=history_text or "（无）",
            user_input=user_input,
        )
        self.rewrite_call_count += 1
        try:
            rewritten = str(self._llm.generate(prompt)).strip()
        except LLMFormatError:
            return user_input
        return rewritten or user_input

    def _compose_run_task(
        self,
        state: SessionState,
        rewritten: str,
        card: TaskCard | None = None,
        mode: IntentMode = "agent",
    ) -> str:
        """needs_history 才注入；类型只决定注入形态。"""
        active = card or default_card(mode, rewritten)
        if not active.needs_history:
            return rewritten
        if active.type == "qa":
            summary = state.rolling_summary.strip()
            if not summary:
                return rewritten
            return (
                f"以下是本会话已知信息：\n会话摘要：{summary}\n\n请回答：{rewritten}"
            )
        return self._compose_budgeted(state, rewritten)

    def _compose_budgeted(self, state: SessionState, rewritten: str) -> str:
        """agent / writer 承接：滚动摘要 + 轮次摘要，不含工具步骤细节。"""
        budget = max(1, self._config.session_summary_limit)
        blocks: list[str] = []
        used = 0
        summary = state.rolling_summary.strip()
        if summary:
            piece = f"会话摘要：{summary}"
            blocks.append(piece)
            used += len(piece)
        for turn in state.turns:
            digest = (
                f"用户：{turn.query_rewritten}\n助手：{_digest_answer(turn.answer)}"
            )
            extra = len(digest) + (1 if blocks else 0)
            if used + extra > budget:
                break
            blocks.append(digest)
            used += extra
        if not blocks:
            return rewritten
        known = "\n".join(blocks)
        if len(known) > budget:
            known = known[:budget]
        return f"以下是本会话已知信息：\n{known}\n\n请回答：{rewritten}"

    def _history_text(self, state: SessionState) -> str:
        parts: list[str] = []
        if state.rolling_summary.strip():
            parts.append(f"摘要：{state.rolling_summary.strip()}")
        for turn in state.turns:
            parts.append(f"用户：{turn.query_rewritten}\n助手：{turn.answer}")
        return "\n".join(parts)

    def _sync_write_ops(self, state: SessionState, steps: list[StepRecord]) -> None:
        registry = self._runner._registry
        for step in steps:
            if step.tool_name == "(format_error)":
                continue
            try:
                tool = registry.get(step.tool_name)
            except KeyError:
                continue
            if not tool.cacheable:
                state.write_op_ledger.seed(step.tool_name, step.args, step.result)

    def _maybe_roll_summary(self, state: SessionState) -> None:
        limit = self._config.session_summary_limit
        keep = self._config.keep_full_turns
        while len(state.turns) > keep:
            oldest = state.turns.pop(0)
            turns_text = (
                f"用户：{oldest.query_rewritten}\n助手：{oldest.answer}"
            )
            if state.rolling_summary.strip():
                turns_text = (
                    f"既有摘要：{state.rolling_summary}\n\n新并入：\n{turns_text}"
                )
            prompt = ROLLING_SUMMARY.format(limit=limit, turns_text=turns_text)
            self.summary_call_count += 1
            try:
                summary = str(self._llm.generate(prompt)).strip()
            except LLMFormatError:
                summary = turns_text[:limit]
            if len(summary) > limit:
                summary = summary[:limit]
            state.rolling_summary = summary
