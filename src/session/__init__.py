from __future__ import annotations

from dataclasses import dataclass, field

from pocket_agent.config import AgentConfig
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import (
    REWRITE_QUERY,
    ROLLING_SUMMARY,
    SESSION_TITLE,
)
from pocket_agent.runner import AgentRunner, EventEmitter, StepRecord


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


class SessionManager:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClientProtocol,
        runner: AgentRunner,
        events: EventEmitter | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._runner = runner
        self._events = events
        self._sessions: dict[str, SessionState] = {}
        # 测试可观测字段
        self.last_rewritten: str = ""
        self.last_run_task: str = ""
        self.last_result = None
        self.rewrite_call_count: int = 0
        self.summary_call_count: int = 0

    def get_state(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> list[dict]:
        """供 Web UI 列举会话。"""
        return [
            {
                "session_id": state.session_id,
                "title": state.title,
                "turns": len(state.turns),
                "rolling_summary": state.rolling_summary,
            }
            for state in self._sessions.values()
        ]

    def chat(self, session_id: str, user_input: str) -> str:
        """多轮对话入口。见文档 07 §1。"""
        state = self.get_state(session_id)
        rewritten = self._rewrite_if_needed(state, user_input)
        self.last_rewritten = rewritten

        run_task = self._compose_run_task(state, rewritten)
        self.last_run_task = run_task

        turn_ledger = CallLedger()
        turn_ledger.merge_records(state.write_op_ledger)
        result = self._runner.run(
            run_task,
            ledger=turn_ledger,
            session_id=session_id,
        )
        self.last_result = result
        self._sync_write_ops(state, result.steps)

        turn = TurnRecord(
            turn=len(state.turns) + 1,
            query_raw=user_input,
            query_rewritten=rewritten,
            answer=str(result.answer),
        )
        state.turns.append(turn)
        self._maybe_rename(state, user_input)
        self._maybe_roll_summary(state)
        return result.answer

    def _maybe_rename(self, state: SessionState, user_input: str) -> None:
        """首轮问答完成后自动生成会话标题；失败回退为用户输入前 12 字截断。

        仅在接入事件总线时触发——单元测试无 events 时不消耗 MockLLM 队列。
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

    def _rewrite_if_needed(self, state: SessionState, user_input: str) -> str:
        if not state.turns and not state.rolling_summary:
            return user_input
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

    def _compose_run_task(self, state: SessionState, rewritten: str) -> str:
        """把会话记忆注入当轮任务，保证 needle 类信息可被回答。"""
        blocks: list[str] = []
        if state.rolling_summary.strip():
            blocks.append(f"会话摘要：{state.rolling_summary.strip()}")
        for turn in state.turns:
            blocks.append(f"用户：{turn.query_rewritten}\n助手：{turn.answer}")
        if not blocks:
            return rewritten
        known = "\n".join(blocks)
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
