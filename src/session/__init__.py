from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pocket_agent.agents import DEFAULT_AGENT, AgentService
from pocket_agent.compressor import ContextCompressor
from pocket_agent.config import AgentConfig
from pocket_agent.intent_router import IntentMode, IntentRouter, has_tool_hint
from pocket_agent.knowledge import (
    EMPTY_RESULT,
    HONEST_PHRASES,
    KnowledgeBase,
    attach_sources,
    extract_sources,
)
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import (
    EMPTY_THINK_NOTE,
    LLMAbortedError,
    LLMClientProtocol,
    LLMFormatError,
    last_usage_of,
)
from pocket_agent.orchestrator import Orchestrator
from pocket_agent.prompts import (
    FINAL_ANSWER,
    FINAL_ANSWER_QA,
    FINAL_ANSWER_QA_AUGMENT,
    REWRITE_QUERY,
    ROLLING_SUMMARY,
    SESSION_TITLE,
)
from pocket_agent.runner import AgentRunner, EventEmitter, RunResult, StepRecord, StopReason
from pocket_agent.task import (
    DEFAULT_WRITER_PLAN,
    Planner,
    TaskCard,
    default_card,
    infer_knowledge_mode,
)
from pocket_agent.verifier import honest_missing_data_note, strip_empty_links, verify
from pocket_agent.workspace import WorkspaceManager

_LLM_CALL_PREVIEW = 120
_DELTA_FLUSH_SEC = 0.040
_DELTA_FLUSH_CHARS = 32

_SID_SAFE = re.compile(r"[^\w.\-]+")
_PIN_MAX = 80
_PIN_BAD = frozenset({"什么", "谁", "哪", "多少", "几岁", "哪个", "哪些"})
_PIN_NAME_RE = re.compile(
    r"(?:我(?:的名字是|叫)|名字是)\s*([^\s，。,.？?！!：:]{1,20})"
)
_PIN_CITY_RE = re.compile(r"我住在\s*([^\s，。,.？?！!]{1,20})")
_PIN_COLOR_RE = re.compile(r"我最喜欢的颜色是\s*([^\s，。,.？?！!]{1,20})")
_PIN_AGE_RE = re.compile(r"我今年\s*(\d+)\s*岁")
_PIN_PASS_RE = re.compile(r"(?:数字)?口令[：:是]?\s*(\d+)")
_PIN_CODE_RE = re.compile(
    r"(?:项目(?:代号|现在叫|原来叫|叫)|(?:后来)?改名为)[：:是]?\s*"
    r"([^\s，。,.？?！!]{1,20})"
)
_PIN_SLOT_PREFIX: tuple[tuple[str, str], ...] = (
    ("name", "我的名字是"),
    ("age", "我今年"),
    ("city", "我住在"),
    ("color", "我最喜欢的颜色是"),
    ("passcode", "口令是"),
    ("codename", "项目代号是"),
)

_DEICTIC_RE = re.compile(
    r"(这两|这两个|这两座|这两件|这几|那个|它们|他们|她们|"
    r"一共|总共|之前|刚才|上面|上述|改名后|现在.*叫|"
    r"温差|相差|比较一下|比起来)"
)
_PRONOUN_RE = re.compile(r"(?:^|[，。,.\s])(?:它|他|她)")
_GROUNDING_NOTE = (
    "回答要求：只能依据对话历史中的具体信息作答；"
    "历史中有的数据必须直接使用，禁止说\"通常\"\"取决于\"\"建议查询\"这类回避性措辞；"
    "信息确实没有时才说明不知道。"
)
_HISTORY_MARKERS = ("以下是本会话已知信息：", "用户：")


def has_deictic_reference(text: str) -> bool:
    """输入含指代/汇总性词时，有历史就必须 rewrite。"""
    raw = text or ""
    return bool(_DEICTIC_RE.search(raw) or _PRONOUN_RE.search(raw))


def _has_history_block(text: str) -> bool:
    return any(marker in (text or "") for marker in _HISTORY_MARKERS)


def _safe_sid(session_id: str) -> str:
    return _SID_SAFE.sub("_", session_id.strip()) or "default"


class WriterProtocol(Protocol):
    def write(self, topic: str, **kwargs: Any) -> str: ...


class _DeltaPump:
    """合并 answer.delta，按 40ms 或 32 字冲刷。"""

    def __init__(
        self,
        events: EventEmitter | None,
        session_id: str,
        turn: int | None,
    ) -> None:
        self._events = events
        self._session_id = session_id
        self._turn = turn
        self._buf = ""
        self._last = time.monotonic()

    def feed(self, text: str) -> None:
        if not text:
            return
        self._buf += text
        now = time.monotonic()
        if now - self._last >= _DELTA_FLUSH_SEC or len(self._buf) >= _DELTA_FLUSH_CHARS:
            self._flush()
            self._last = now

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._buf or self._events is None:
            self._buf = ""
            return
        payload: dict[str, Any] = {
            "session_id": self._session_id,
            "turn": self._turn,
            "delta": self._buf,
        }
        self._events.emit("answer.delta", payload)
        self._buf = ""


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
    fact_pins: list[str] = field(default_factory=list)
    write_op_ledger: CallLedger = field(default_factory=CallLedger)
    title: str = "新会话"
    agent: str = DEFAULT_AGENT
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
        orchestrator: Orchestrator | None = None,
        knowledge: KnowledgeBase | None = None,
        agents: AgentService | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._runner = runner
        self._compressor = ContextCompressor(config, llm)
        self._events = events
        self._intent_router = intent_router or IntentRouter(llm)
        self._planner = Planner(llm)
        self._writer = writer
        self._workspace = workspace
        self._orchestrator = orchestrator
        self._knowledge = knowledge
        self._agents = agents
        self._sessions_dir = Path(sessions_dir) if sessions_dir else None
        self._active_agent = DEFAULT_AGENT
        self._last_system = ""
        self._sessions: dict[str, SessionState] = {}
        if self.persist_enabled:
            self._load_all()
        # 测试可观测字段
        self.last_rewritten: str = ""
        self.last_run_task: str = ""
        self.last_result = None
        self.last_intent: IntentMode | None = None
        self.last_task_card: TaskCard | None = None
        self.last_verify_reasons: list[str] = []
        self.last_evidence: str = ""
        self.rewrite_call_count: int = 0
        self.summary_call_count: int = 0
        self._active_turn: int | None = None
        self._active_session_id: str = ""
        self._answer_done_emitted: bool = False

    @property
    def persist_enabled(self) -> bool:
        return self._agents is not None or self._sessions_dir is not None

    def _store_key(self, session_id: str, agent: str | None = None) -> str:
        if self._agents is None:
            return session_id
        return f"{agent or self._active_agent}\x1f{session_id}"

    def get_state(self, session_id: str, *, agent: str | None = None) -> SessionState:
        key = self._store_key(session_id, agent)
        if key not in self._sessions:
            self._sessions[key] = SessionState(
                session_id=session_id,
                agent=agent or self._active_agent,
            )
        return self._sessions[key]

    def list_sessions(self, agent: str | None = None) -> list[dict]:
        """供 Web UI 列举会话，按 updated_at 降序。"""
        rows = [
            {
                "session_id": state.session_id,
                "agent": state.agent,
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
            if agent is None or state.agent == agent
        ]
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        return rows

    def delete_session(self, session_id: str, *, agent: str | None = None) -> bool:
        """删除会话（内存 + 落盘文件）。"""
        key = self._store_key(session_id, agent)
        state = self._sessions.pop(key, None)
        path = self._session_path(session_id, (state.agent if state else None) or agent)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def _session_dir(self, agent: str | None) -> Path | None:
        if self._agents is not None:
            try:
                return self._agents.sessions_dir(agent or self._active_agent)
            except ValueError:
                return self._agents.sessions_dir(DEFAULT_AGENT)
        if self._sessions_dir is None:
            return None
        return self._sessions_dir

    def _session_path(self, session_id: str, agent: str | None = None) -> Path | None:
        folder = self._session_dir(agent)
        if folder is None:
            return None
        return folder / f"{_safe_sid(session_id)}.json"

    def _save_session(self, state: SessionState) -> None:
        """原子写入会话到 agent 目录或兼容的 sessions_dir。"""
        folder = self._session_dir(state.agent)
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": state.session_id,
            "agent": state.agent,
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
            "fact_pins": list(state.fact_pins),
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }
        path = folder / f"{_safe_sid(state.session_id)}.json"
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
        """启动时扫描各 agent sessions/ 或兼容目录；损坏文件跳过。"""
        if self._agents is not None:
            for profile in self._agents.list():
                self._load_dir(self._agents.sessions_dir(profile.name), profile.name)
            self._load_dir(self._agents.shared_dir() / "sessions", "shared")
            return
        if self._sessions_dir is not None:
            self._load_dir(self._sessions_dir, DEFAULT_AGENT)

    def _load_dir(self, folder: Path, agent: str) -> None:
        if not folder.is_dir():
            return
        for path in folder.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or "session_id" not in data:
                continue
            sid = str(data["session_id"])
            owner = str(data.get("agent") or agent)
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
            raw_pins = data.get("fact_pins") or []
            pins = [
                str(item).strip()[:_PIN_MAX]
                for item in raw_pins
                if isinstance(item, str) and item.strip()
            ]
            state = SessionState(
                session_id=sid,
                agent=owner,
                turns=turns,
                rolling_summary=str(data.get("rolling_summary", "")),
                fact_pins=pins,
                title=str(data.get("title", "新会话")),
                created_at=float(data.get("created_at", time.time())),
                updated_at=float(data.get("updated_at", time.time())),
            )
            self._sessions[self._store_key(sid, owner)] = state

    def chat(
        self,
        session_id: str,
        user_input: str,
        *,
        force_mode: IntentMode | None = None,
        permission: str = "workspace",
        workspace_id: str | None = None,
        agent: str = DEFAULT_AGENT,
    ) -> str:
        """多轮对话入口。见文档 07 §1。

        force_mode：用户纠正意图时传入，跳过自动分类。
        permission：readonly / workspace / full，映射审批策略。
        workspace_id：绑定工作区（None 则不变）。
        agent：当前 Agent 名称，决定会话目录与人格。
        """
        self._active_agent = agent or DEFAULT_AGENT
        self._active_session_id = session_id
        self._answer_done_emitted = False
        self._last_system = self._system_preamble(user_input)
        state = self.get_state(session_id, agent=self._active_agent)
        state.agent = self._active_agent
        state.fact_pins = apply_fact_pins(state.fact_pins, user_input)
        if self._agents is not None and permission == "workspace":
            try:
                permission = self._agents.get(self._active_agent).permission
            except KeyError:
                pass
        self._bind_agent_workspace(session_id, workspace_id)
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
            self._active_turn = len(state.turns) + 1
            card, mode, confidence, source, reason = self._resolve_card(
                user_input, state.rolling_summary
            )
        self.last_intent = mode
        self.last_task_card = card
        self.last_evidence = ""
        turn_no = len(state.turns) + 1
        self._active_turn = turn_no
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
            needs_rewrite = card.needs_history or (
                bool(state.turns) and has_deictic_reference(user_input)
            )
            if needs_rewrite and card.type == "agent":
                rewritten = self._rewrite_if_needed(state, user_input)
            elif needs_rewrite and card.type == "qa":
                rewritten = self._rewrite_if_needed(
                    state, user_input, summary_only=True
                )
            else:
                rewritten = user_input
        self.last_rewritten = rewritten

        run_task = self._compose_run_task(state, rewritten, card=card)
        self.last_run_task = run_task

        answer = self._dispatch(card, run_task, session_id, permission, state)
        answer = self._verify_answer(
            answer, card, run_task, session_id, user_input=user_input
        )
        if card.type == "qa" and not self._answer_done_emitted:
            self._emit_answer_done(answer, stop_reason="qa")

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
                t0 = time.perf_counter()
                planned = self._planner.plan(
                    user_input,
                    rolling_summary,
                    system=self._last_system,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                raw = getattr(self._planner, "last_raw", None)
                response = json.dumps(
                    raw if isinstance(raw, dict) else planned.to_dict(),
                    ensure_ascii=False,
                )
                self._emit_llm_call(
                    session_id=self._active_session_id,
                    purpose="planner",
                    prompt=getattr(self._planner, "last_prompt", "") or "",
                    response=response,
                    elapsed_ms=elapsed_ms,
                )
                card = self._ground_card(planned, user_input)
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
        kmode = infer_knowledge_mode(user_input)
        needs_k = kmode != "none"
        if (
            new_type == card.type
            and plan == list(card.plan)
            and needs_k == card.needs_knowledge
            and kmode == card.knowledge_mode
        ):
            return card
        return TaskCard(
            type=new_type,
            goal=card.goal,
            constraints=card.constraints,
            plan=plan,
            needs_history=card.needs_history or new_type == "agent",
            needs_knowledge=needs_k,
            knowledge_mode=kmode,
        )

    def _dispatch(
        self,
        card: TaskCard,
        run_task: str,
        session_id: str,
        permission: str,
        state: SessionState,
    ) -> str:
        if card.type == "qa":
            return self._run_qa(
                card,
                run_task,
                session_id,
                query=self.last_rewritten,
                has_history=bool(state.turns),
            )
        if self._orchestrator is not None and self._orchestrator.should_orchestrate(card):
            return self._run_orchestrated(
                card, run_task, session_id, permission, state
            )
        if card.type == "writer" and self._writer is not None:
            return self._run_writer(card, run_task, session_id)
        turn_ledger = CallLedger()
        turn_ledger.merge_records(state.write_op_ledger)
        result = self._runner.run(
            self._agent_task(card, run_task),
            ledger=turn_ledger,
            session_id=session_id,
            permission=permission,
            turn=self._active_turn,
        )
        self.last_result = result
        self._remember_tool_evidence(result.steps)
        self._sync_write_ops(state, result.steps)
        self._mark_plan_progress(session_id, len(result.steps))
        return str(result.answer)

    def _run_orchestrated(
        self,
        card: TaskCard,
        run_task: str,
        session_id: str,
        permission: str,
        state: SessionState,
    ) -> str:
        orch = self._orchestrator
        if orch is None:
            result = self._runner.run(
                self._agent_task(card, run_task),
                session_id=session_id,
                permission=permission,
                turn=self._active_turn,
            )
            self.last_result = result
            return str(result.answer)
        outcome = orch.run(
            card,
            self._agent_task(card, run_task),
            session_id=session_id,
            permission=permission,
            turn=self._active_turn,
        )
        self._remember_tool_evidence(outcome.steps)
        stop = (
            StopReason.STOPPED
            if outcome.stop_reason in {"aborted", "stopped"}
            else StopReason.MODEL_DONE
        )
        self.last_result = RunResult(
            answer=outcome.answer,
            steps=outcome.steps,
            stop_reason=stop,
            stats=dict(outcome.stats),
        )
        self._sync_write_ops(state, outcome.steps)
        return str(outcome.answer)

    def _run_writer(self, card: TaskCard, topic: str, session_id: str) -> str:
        writer = self._writer
        if writer is None:
            return topic
        pump: _DeltaPump | None = None
        if self._events is not None and hasattr(writer, "on_thinking"):
            def _emit_think(text: str) -> None:
                if text and self._config.reasoning_enabled and self._events is not None:
                    self._events.emit(
                        "step.thinking",
                        {
                            "text": text,
                            "session_id": session_id,
                            "turn": self._active_turn,
                        },
                    )

            writer.on_thinking = _emit_think
        if self._events is not None:
            pump = _DeltaPump(self._events, session_id, self._active_turn)
            if hasattr(writer, "on_delta"):
                writer.on_delta = pump.feed
        if self._events is not None and hasattr(writer, "on_section"):
            def _on_section(index: int, _section: object, _body: str) -> None:
                if self._events is not None:
                    self._events.emit(
                        "task.step_done",
                        {
                            "session_id": session_id,
                            "index": max(0, index - 1),
                            "turn": self._active_turn,
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
        material = ""
        if card.needs_knowledge:
            material = self._retrieve(goal)
        try:
            answer = writer.write(
                goal,
                total_chars=total_chars,
                material=material,
                plan=list(card.plan) or None,
            )
        except LLMAbortedError:
            answer = "\n\n".join(writer.last_sections)
            writer.stopped = True
        finally:
            if pump is not None:
                pump.close()
            if hasattr(writer, "on_delta"):
                writer.on_delta = None
        self.last_result = None
        reason = "stopped" if getattr(writer, "stopped", False) else "writer"
        self._emit_answer_done(answer, stop_reason=reason)
        return answer

    def _run_qa(
        self,
        card: TaskCard,
        run_task: str,
        session_id: str = "",
        *,
        query: str = "",
        has_history: bool = False,
    ) -> str:
        prompt = run_task
        search_query = (query or "").strip()
        if card.needs_knowledge:
            evidence = self._retrieve(search_query)
            if _is_empty_retrieval(evidence):
                if card.knowledge_mode != "augment":
                    answer = EMPTY_RESULT
                    self.last_result = None
                    self._emit_answer_done(answer, stop_reason="qa")
                    return answer
                if self._events is not None:
                    self._events.emit(
                        "rag.empty",
                        {
                            "session_id": session_id,
                            "turn": self._active_turn,
                            "query": search_query,
                        },
                    )
                if not has_history:
                    prompt = FINAL_ANSWER_QA.format(
                        task=run_task, evidence=evidence
                    )
            elif has_history and card.knowledge_mode == "augment":
                prompt = FINAL_ANSWER_QA_AUGMENT.format(
                    task=run_task, evidence=evidence
                )
            else:
                prompt = FINAL_ANSWER_QA.format(task=run_task, evidence=evidence)
        if has_history and _has_history_block(run_task):
            prompt = f"{prompt.rstrip()}\n{_GROUNDING_NOTE}"
        prompt = self._wrap_system(prompt)
        try:
            answer = self._generate_user_text(prompt, session_id)
        except LLMAbortedError:
            answer = "（已停止）"
            self.last_result = None
            self._emit_answer_done(answer, stop_reason="stopped")
            return answer
        except LLMFormatError:
            answer = EMPTY_THINK_NOTE
        self.last_result = None
        return answer

    def _generate_user_text(self, prompt: str, session_id: str) -> str:
        stream = getattr(self._llm, "generate_stream", None)
        if callable(stream) and self._events is not None:
            pump = _DeltaPump(self._events, session_id, self._active_turn)
            try:
                return str(stream(prompt, pump.feed)).strip()
            finally:
                pump.close()
        return str(self._llm.generate(prompt)).strip()

    def _retrieve(self, query: str) -> str:
        if self._knowledge is None:
            self.last_evidence = EMPTY_RESULT
            return self.last_evidence
        text = self._knowledge.search_text(query)
        self.last_evidence = text
        return text

    def _remember_tool_evidence(self, steps: list[StepRecord]) -> None:
        lines: list[str] = []
        for step in steps:
            if step.tool_name in {"search_knowledge", "wiki_lookup", "web_search"}:
                lines.append(str(step.result or ""))
        if lines:
            self.last_evidence = "\n".join(lines)

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
        *,
        user_input: str = "",
    ) -> str:
        self.last_verify_reasons = []
        evidence = self.last_evidence
        answer = strip_empty_links(answer)
        if not self._config.verifier_enabled:
            return self._attach_sources(answer, card, evidence, user_input=user_input)
        real_calls: int | None = None
        if self._runner._registry.all():
            real_calls = 0
            if self.last_result is not None:
                real_calls = int(self.last_result.stats.get("real_calls") or 0)
        hint_text = user_input
        if self.last_rewritten and self.last_rewritten != user_input:
            hint_text = f"{user_input}\n{self.last_rewritten}"
        tool_names = self._result_tool_names()
        reasons = verify(
            answer,
            card,
            evidence=evidence,
            user_input=hint_text,
            real_calls=real_calls,
            tool_names=tool_names,
        )
        retries = max(0, int(self._config.answer_max_retries))
        if card.type == "writer":
            retries = 0
        attempt = 0
        while reasons and attempt < retries:
            self.last_verify_reasons = list(reasons)
            self._emit_rejected(session_id, reasons)
            history_text = self._tool_result_text() or evidence or "（无）"
            prompt = FINAL_ANSWER.format(task=run_task, history_text=history_text)
            restated = f"\n用户原话要求：{user_input}" if user_input.strip() else ""
            missing = next((r for r in reasons if r.startswith("领域工具缺失")), "")
            domain_note = f"\n{missing}，请改用对应领域工具获取数据后再回答。" if missing else ""
            prompt = self._wrap_system(
                f"{prompt}\n\n验收未通过：{'；'.join(reasons)}{restated}{domain_note}\n"
                "请根据已有结果直接回答用户，不要复述执行过程。"
            )
            try:
                t0 = time.perf_counter()
                retry_text = self._generate_user_text(prompt, session_id)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._emit_llm_call(
                    session_id=session_id,
                    purpose="verify_retry",
                    prompt=prompt,
                    response=retry_text,
                    elapsed_ms=elapsed_ms,
                )
                answer = strip_empty_links(retry_text or answer)
            except LLMAbortedError:
                break
            except LLMFormatError:
                break
            reasons = verify(
                answer,
                card,
                evidence=evidence,
                user_input=hint_text,
                real_calls=real_calls,
                tool_names=tool_names,
            )
            attempt += 1
        if reasons:
            self.last_verify_reasons = list(reasons)
            self._emit_rejected(session_id, reasons)
            cleaned = strip_empty_links(
                self._fallback_answer(answer, reasons, user_input=user_input)
            )
            return self._attach_sources(
                cleaned, card, evidence, user_input=user_input
            )
        return self._attach_sources(answer, card, evidence, user_input=user_input)

    def _result_tool_names(self) -> list[str] | None:
        result = self.last_result
        if result is None:
            return None
        return [
            s.tool_name
            for s in result.steps
            if s.tool_name and s.tool_name not in ("(format_error)", "(domain_gate)")
        ]

    def _tool_result_text(self) -> str:
        result = self.last_result
        if result is None:
            return ""
        parts = [
            str(step.result)
            for step in result.steps
            if step.result and not step.error
        ]
        return "\n".join(parts).strip()

    def _fallback_answer(
        self,
        answer: str,
        reasons: list[str],
        *,
        user_input: str = "",
    ) -> str:
        missing_note = honest_missing_data_note(user_input, reasons)
        if missing_note:
            return missing_note
        if "格式违反" in reasons:
            blob = self._tool_result_text()
            match = re.search(r"-?\d+", blob)
            if match:
                return match.group(0)
        if "正式回答为空" in reasons:
            parts = self._tool_result_text()
            if parts:
                return parts
            body = (answer or "").replace(EMPTY_THINK_NOTE, "").strip()
            for marker in ("[生成回答失败]", "[格式错误]"):
                body = body.replace(marker, "")
            return body.strip() or "未能生成正式回答"
        return answer or ""

    def _attach_sources(
        self,
        answer: str,
        card: TaskCard,
        evidence: str,
        *,
        user_input: str = "",
    ) -> str:
        if not card.needs_knowledge:
            return answer
        return attach_sources(
            answer, extract_sources(evidence), user_input=user_input
        )

    def _emit_rejected(self, session_id: str, reasons: list[str]) -> None:
        if self._events is not None:
            payload: dict[str, Any] = {
                "session_id": session_id,
                "reasons": reasons,
            }
            if self._active_turn is not None:
                payload.setdefault("turn", self._active_turn)
            self._events.emit("answer.rejected", payload)

    def _emit_answer_done(self, answer: str, *, stop_reason: str) -> None:
        if self._events is None:
            return
        meta = getattr(self, "_run_meta", None)
        extra = dict(meta) if isinstance(meta, dict) else {}
        self._answer_done_emitted = True
        payload: dict[str, Any] = {
            "answer": answer,
            "stop_reason": stop_reason,
            "session_id": self._active_session_id,
            "total_steps": 0,
            "llm_ms": 0,
            "tool_ms": 0,
            "cache_hit_rate": 0.0,
            "real_calls": 0,
            "intercepts": 0,
            "prompt_tokens_max": 0,
            "elapsed_ms": 0,
            **extra,
        }
        if self._active_turn is not None:
            payload.setdefault("turn", self._active_turn)
        usage = last_usage_of(self._llm)
        if usage is not None:
            payload["usage"] = usage
        self._events.emit("answer.done", payload)

    def _emit_llm_call(
        self,
        *,
        session_id: str,
        purpose: str,
        prompt: str,
        response: str,
        elapsed_ms: float,
    ) -> None:
        if self._events is None:
            return
        payload: dict[str, Any] = {
            "session_id": session_id,
            "purpose": purpose,
            "prompt_preview": prompt[:_LLM_CALL_PREVIEW],
            "response_preview": response[:_LLM_CALL_PREVIEW],
            "elapsed_ms": int(elapsed_ms),
        }
        if self._active_turn is not None:
            payload["turn"] = self._active_turn
        usage = last_usage_of(self._llm)
        if usage is not None:
            payload["usage"] = usage
        self._events.emit("llm.call", payload)

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
            t0 = time.perf_counter()
            raw = str(self._llm.generate(prompt)).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._emit_llm_call(
                session_id=state.session_id,
                purpose="rename",
                prompt=prompt,
                response=raw,
                elapsed_ms=elapsed_ms,
            )
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
            history_text = (
                f"摘要：{summary}" if summary else self._history_text(state)
            )
            if not history_text.strip():
                return user_input
        else:
            history_text = self._history_text(state)
        prompt = REWRITE_QUERY.format(
            history_text=history_text or "（无）",
            user_input=user_input,
        )
        self.rewrite_call_count += 1
        try:
            t0 = time.perf_counter()
            rewritten = str(self._llm.generate(prompt)).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._emit_llm_call(
                session_id=state.session_id,
                purpose="rewrite",
                prompt=prompt,
                response=rewritten,
                elapsed_ms=elapsed_ms,
            )
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
        """qa/agent 有历史则硬注入；needs_history 只决定是否 rewrite。"""
        active = card or default_card(mode, rewritten)
        if active.type == "writer" or (not state.turns and not state.fact_pins):
            body = rewritten
        else:
            body = self._compose_budgeted(state, rewritten)
        prefix = self._context_prefix()
        if not prefix:
            return body
        return f"{prefix}\n\n{body}"

    def _compose_budgeted(self, state: SessionState, rewritten: str) -> str:
        """滚动摘要 + 最近 keep_full_turns 轮原文；超预算先砍旧轮。"""
        keep = max(1, self._config.keep_full_turns)
        recent = state.turns[-keep:]
        pairs = [(turn.query_rewritten, turn.answer) for turn in recent]
        return self._compressor.pack_session_history(
            summary=state.rolling_summary,
            turns=pairs,
            rewritten=rewritten,
            fact_pins=list(state.fact_pins),
        )

    def _history_text(self, state: SessionState) -> str:
        parts: list[str] = []
        if state.fact_pins:
            parts.append("长期事实：" + "；".join(state.fact_pins))
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
                t0 = time.perf_counter()
                summary = str(self._llm.generate(prompt)).strip()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._emit_llm_call(
                    session_id=state.session_id,
                    purpose="summarize",
                    prompt=prompt,
                    response=summary,
                    elapsed_ms=elapsed_ms,
                )
            except LLMFormatError:
                summary = turns_text[:limit]
            if len(summary) > limit:
                summary = summary[:limit]
            state.rolling_summary = summary

    def _system_preamble(self, task_text: str) -> str:
        if self._agents is None:
            return ""
        try:
            return self._agents.system_preamble(self._active_agent, task_text)
        except (KeyError, ValueError):
            return ""

    def _wrap_system(self, prompt: str) -> str:
        if not self._last_system:
            return prompt
        return f"{self._last_system}\n\n{prompt}"

    def _context_prefix(self) -> str:
        """实体槽位永不进入滚动摘要，每轮单独注入。"""
        if self._agents is None:
            return ""
        try:
            return self._agents.format_entities(self._active_agent)
        except (KeyError, ValueError):
            return ""

    def _bind_agent_workspace(self, session_id: str, workspace_id: str | None) -> None:
        if self._workspace is None:
            return
        if workspace_id is not None:
            return
        if self._agents is None:
            return
        try:
            root = self._agents.workspace_dir(self._active_agent)
            shared_files = self._agents.shared_dir() / "files"
        except (KeyError, ValueError):
            return
        wid = f"agent:{self._active_agent}"
        self._workspace.ensure(wid, self._active_agent, str(root))
        if hasattr(self._workspace, "add_readonly_root"):
            self._workspace.add_readonly_root(str(shared_files))
        self._workspace.bind_session(session_id, wid)


def _is_empty_retrieval(text: str) -> bool:
    body = (text or "").strip()
    return (not body) or any(phrase in body for phrase in HONEST_PHRASES)


def apply_fact_pins(existing: list[str], user_input: str) -> list[str]:
    """合并用户原文抽出的事实钉：按槽位覆盖，去重，每条 ≤80 字。"""
    keyed: dict[str, str] = {}
    extra: list[str] = []
    for pin in existing:
        text = str(pin).strip()[:_PIN_MAX]
        if not text:
            continue
        slot = _slot_of_pin(text)
        if slot:
            keyed[slot] = text
        elif text not in extra:
            extra.append(text)
    for slot, stmt in extract_fact_pins(user_input):
        clipped = stmt.strip()[:_PIN_MAX]
        if not clipped:
            continue
        if slot == "extra":
            if clipped not in extra and clipped not in keyed.values():
                extra.append(clipped)
        else:
            keyed[slot] = clipped
    ordered = [
        keyed[name]
        for name, _prefix in _PIN_SLOT_PREFIX
        if name in keyed
    ]
    for item in extra:
        if item not in ordered:
            ordered.append(item)
    return ordered


def extract_fact_pins(user_input: str) -> list[tuple[str, str]]:
    """仅从高置信用户陈述抽出 (slot, statement)。"""
    text = _clip_pin_source(user_input)
    if not text:
        return []
    found: list[tuple[str, str]] = []

    def _add(slot: str, value: str, statement: str) -> None:
        cleaned = value.strip("。．.，,：: ")
        if not cleaned or cleaned in _PIN_BAD or "?" in cleaned or "？" in cleaned:
            return
        found.append((slot, statement))

    match = _PIN_NAME_RE.search(text)
    if match:
        _add("name", match.group(1), f"我的名字是{match.group(1).strip()}")
    match = _PIN_CITY_RE.search(text)
    if match:
        _add("city", match.group(1), f"我住在{match.group(1).strip()}")
    match = _PIN_COLOR_RE.search(text)
    if match:
        _add("color", match.group(1), f"我最喜欢的颜色是{match.group(1).strip()}")
    match = _PIN_AGE_RE.search(text)
    if match:
        _add("age", match.group(1), f"我今年{match.group(1).strip()}岁")
    match = _PIN_PASS_RE.search(text)
    if match:
        _add("passcode", match.group(1), f"口令是{match.group(1).strip()}")
    match = _PIN_CODE_RE.search(text)
    if match:
        _add("codename", match.group(1), f"项目代号是{match.group(1).strip()}")
    return found


def _clip_pin_source(user_input: str) -> str:
    body = (user_input or "").strip()
    body = re.split(r"(?:只需确认|只确认记住|不要复述[^。]*)", body)[0]
    return body.strip()


def _slot_of_pin(statement: str) -> str:
    for slot, prefix in _PIN_SLOT_PREFIX:
        if statement.startswith(prefix):
            return slot
    return ""
