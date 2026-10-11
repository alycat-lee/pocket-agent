from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pocket_agent.config import AgentConfig
from pocket_agent.intent_router import IntentMode, IntentRouter
from pocket_agent.ledger import CallLedger
from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import (
    REWRITE_QUERY,
    ROLLING_SUMMARY,
    SESSION_TITLE,
)
from pocket_agent.runner import AgentRunner, EventEmitter, StepRecord
from pocket_agent.tools import ToolRegistry
from pocket_agent.workspace import WorkspaceManager

_SID_SAFE = re.compile(r"[^\w.\-]+")


def _safe_sid(session_id: str) -> str:
    return _SID_SAFE.sub("_", session_id.strip()) or "default"


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
        # 意图自动路由（仅在接入事件总线时触发——单元测试无 events 时不消耗队列）
        rewritten: str | None = None
        source = "fallback"
        reason = "无事件总线"
        if self._events is None:
            mode: IntentMode = "agent"
            confidence = 1.0
        elif force_mode is not None:
            mode = force_mode
            confidence = 1.0
            source = "manual"
            reason = "用户纠正"
            last = state.turns[-1] if state.turns else None
            if last is not None and last.query_raw == user_input:
                # 纠正重跑：复用当轮指代补全，不重新 rewrite
                rewritten = last.query_rewritten
                state.turns.pop()
        else:
            mode, confidence, source, reason = self._intent_router.classify(
                user_input
            )
        self.last_intent = mode
        turn_no = len(state.turns) + 1
        if self._events is not None:
            # 每轮事件流第一条：intent.detected（先于 workspace/permission/runner）
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
            rewritten = self._rewrite_if_needed(state, user_input)
        self.last_rewritten = rewritten

        run_task = self._compose_run_task(state, rewritten)
        self.last_run_task = run_task

        if mode == "writer" and self._writer is not None:
            answer = self._writer.write(rewritten)
            self.last_result = None
            if self._events is not None:
                self._events.emit(
                    "answer.done",
                    {
                        "answer": answer,
                        "stop_reason": "writer",
                        "total_steps": 0,
                        "llm_ms": 0,
                        "tool_ms": 0,
                        "cache_hit_rate": 0.0,
                        "real_calls": 0,
                        "intercepts": 0,
                        "prompt_tokens_max": 0,
                        "elapsed_ms": 0,
                    },
                )
        else:
            turn_ledger = CallLedger()
            turn_ledger.merge_records(state.write_op_ledger)
            # qa 模式不给工具：空注册表 → recall 返回空 → 直接 final_answer
            run_registry: ToolRegistry | None = None
            if mode == "qa":
                run_registry = ToolRegistry(self._config)
            result = self._runner.run(
                run_task,
                ledger=turn_ledger,
                session_id=session_id,
                registry=run_registry,
                permission=permission,
            )
            self.last_result = result
            self._sync_write_ops(state, result.steps)
            answer = str(result.answer)

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
