"""终端 TUI 界面插件：只消费 ctx 服务与事件总线，不改引擎。"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pocket_agent.agents import DEFAULT_AGENT
from pocket_agent.config import AgentConfig, load_settings, save_settings
from pocket_agent.providers import (
    ProvidersConfig,
    describe_provider,
    load_providers_config,
    providers_to_dict,
)
from pocket_agent.runtime import Context, PluginError

PROMPT_TOOLKIT_AVAILABLE = True
try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import StyleAndTextTuples
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import TextArea
except ImportError:  # pragma: no cover — 装配时再报安装指引
    PROMPT_TOOLKIT_AVAILABLE = False
    Application = Any  # type: ignore[misc,assignment]
    StyleAndTextTuples = list  # type: ignore[misc,assignment]
    KeyBindings = Any  # type: ignore[misc,assignment]
    Dimension = Any  # type: ignore[misc,assignment]
    HSplit = Any  # type: ignore[misc,assignment]
    Layout = Any  # type: ignore[misc,assignment]
    Window = Any  # type: ignore[misc,assignment]
    FormattedTextControl = Any  # type: ignore[misc,assignment]
    Style = Any  # type: ignore[misc,assignment]
    TextArea = Any  # type: ignore[misc,assignment]

_PT_MISSING = (
    "TUI 需要 prompt_toolkit：pip install 'pocket-agent[tui]' 或 pip install prompt_toolkit"
)
_APPROVAL_TIMEOUT = 60.0
_SLASH_NAMES = (
    "help",
    "new",
    "sessions",
    "resume",
    "agent",
    "model",
    "export",
    "stop",
    "quit",
)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_SID_SAFE = re.compile(r"[^\w.\-]+")


@dataclass(frozen=True)
class SlashCommand:
    name: str
    arg: str


def parse_slash(text: str) -> SlashCommand | None:
    """输入以 / 开头则解析为斜杠命令；否则 None（交给模型）。"""
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    body = raw[1:].strip()
    if not body:
        return SlashCommand("help", "")
    parts = body.split(None, 1)
    return SlashCommand(parts[0].lower(), parts[1].strip() if len(parts) > 1 else "")


def slash_help_text() -> str:
    return (
        "/help 列出全部命令\n"
        "/new 新会话\n"
        "/sessions 列出当前 agent 会话\n"
        "/resume <序号|id> 恢复会话\n"
        "/agent [名] 切换或列出 agent\n"
        "/model [序号] 列出或切换默认模型\n"
        "/export 导出当前会话为 Markdown\n"
        "/stop 停止当前生成\n"
        "/quit 退出"
    )


def unknown_slash_hint(name: str) -> str:
    return f"未知命令 /{name}，输入 /help 查看"


def args_summary(args: object, limit: int = 80) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    text = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def project_event(event: str, payload: dict[str, Any]) -> list[str]:
    """事件 → 终端投影行（只读，不改状态机）。"""
    data = payload or {}
    if event == "intent.detected":
        mode = data.get("mode") or "?"
        source = data.get("source") or "?"
        return [f"意图：{mode}（{source}）"]
    if event == "task.planned":
        lines = [f"任务卡：{data.get('goal') or ''}"]
        plan = data.get("plan")
        if isinstance(plan, list):
            for index, step in enumerate(plan, start=1):
                if str(step).strip():
                    lines.append(f"  {index}. {step}")
        return lines
    if event == "step.tool_call":
        name = str(data.get("tool") or data.get("current_tool") or data.get("name") or "")
        mark = "✖" if data.get("error") else "✓"
        extra = args_summary(data.get("args"))
        line = f"● 工具 {name} {extra} {mark}".strip()
        return [" ".join(line.split())]
    if event == "step.intercepted":
        return [f"已拦截：{data.get('reason') or ''}"]
    if event == "rag.empty":
        return ["未检索到资料，由模型自身作答"]
    if event == "answer.rejected":
        reasons = data.get("reasons") or []
        if isinstance(reasons, list):
            text = "；".join(str(item) for item in reasons if str(item).strip())
        else:
            text = str(reasons)
        return [f"验收未通过：{text}，正在重试"]
    return []


def format_settlement(payload: dict[str, Any]) -> str:
    turn = payload.get("turn") if payload.get("turn") is not None else "?"
    steps = payload.get("total_steps") or 0
    rate = float(payload.get("cache_hit_rate") or 0.0)
    intercepts = payload.get("intercepts") or 0
    reason = payload.get("stop_reason") or ""
    return (
        f"轮次 {turn} · {steps} 步 · 缓存命中 {rate:.0%} · "
        f"拦截 {intercepts} · 停止原因 {reason}"
    ).rstrip()


def format_approval_line(payload: dict[str, Any]) -> str:
    tool = payload.get("tool") or payload.get("tool_name") or "?"
    reason = payload.get("reason") or ""
    summary = args_summary(payload.get("args"))
    return f"审批 · {tool} · {summary} · {reason}".rstrip(" ·")


def resolve_approval_key(key: str | None) -> bool:
    """y 允许；n / Esc / 超时(None) 拒绝。"""
    if key is None:
        return False
    token = key.lower()
    return token == "y"


def prompt_approval(
    payload: dict[str, Any],
    *,
    timeout: float = _APPROVAL_TIMEOUT,
    read_char: Callable[[float], str | None],
    printer: Callable[[str], None] | None = None,
) -> bool:
    """阻塞式审批问答。read_char(timeout) 返回单键或 None（超时）。"""
    emit = printer or print
    emit(format_approval_line(payload))
    key = read_char(timeout)
    if key is None:
        emit("审批超时已拒绝")
        return False
    return resolve_approval_key(key)


def pick_latest_session(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("updated_at") or 0.0))


def new_session_id() -> str:
    return f"t-{int(time.time())}"


def format_session_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["（无会话）"]
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        sid = row.get("session_id") or ""
        title = row.get("title") or sid
        updated = row.get("updated_at")
        stamp = ""
        if isinstance(updated, (int, float)) and updated > 0:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(updated)))
        lines.append(f"{index}. {title}  {sid}  {stamp}".rstrip())
    return lines


def resolve_resume_token(token: str, rows: list[dict[str, Any]]) -> str | None:
    text = (token or "").strip()
    if not text:
        return None
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(rows):
            return str(rows[index - 1].get("session_id") or "")
        return None
    return text


def format_agent_rows(names: list[str], current: str) -> list[str]:
    if not names:
        return ["（无 agent）"]
    lines: list[str] = []
    for name in names:
        mark = "（当前）" if name == current else ""
        lines.append(f"{name}{mark}")
    return lines


def format_provider_rows(cfg: ProvidersConfig) -> list[str]:
    if not cfg.providers:
        return ["（无 provider）"]
    lines: list[str] = []
    for index, item in enumerate(cfg.providers, start=1):
        mark = "（默认）" if item.id == cfg.default_provider_id else ""
        lines.append(f"{index}. {item.name}  {item.model or '-'}{mark}")
    return lines


def apply_default_provider(cfg: ProvidersConfig, index: int) -> tuple[ProvidersConfig | None, str]:
    if index < 1 or index > len(cfg.providers):
        return None, "序号无效，输入 /model 查看列表"
    chosen = cfg.providers[index - 1]
    cfg.default_provider_id = chosen.id
    cfg.routing.choose_fill = chosen.id
    cfg.routing.generate = chosen.id
    return cfg, f"已切换默认模型：{chosen.name} · {chosen.model or '-'}"


def export_markdown(session_id: str, title: str, turns: list[Any]) -> str:
    lines = [f"# {title or session_id}", ""]
    for turn in turns:
        query = getattr(turn, "query_raw", None)
        if query is None and isinstance(turn, dict):
            query = turn.get("query_raw") or turn.get("query") or ""
        answer = getattr(turn, "answer", None)
        if answer is None and isinstance(turn, dict):
            answer = turn.get("answer") or ""
        lines.append(f"## 用户\n\n{query}\n")
        lines.append(f"## 助手\n\n{answer}\n")
    return "\n".join(lines).rstrip() + "\n"


def safe_export_name(session_id: str) -> str:
    return _SID_SAFE.sub("_", session_id.strip()) or "session"


def running_status(payload: dict[str, Any]) -> str:
    step = payload.get("step_index")
    if step is None:
        step = payload.get("step") or "?"
    tool = payload.get("current_tool") or payload.get("tool") or ""
    if tool:
        return f"运行中 · S{step} · {tool}…"
    return f"运行中 · S{step} · …"


def basic_markdown_ft(text: str) -> StyleAndTextTuples:
    """标题 / 粗体 / 代码块的轻量 formatted text，不是完整渲染器。"""
    chunks: StyleAndTextTuples = []
    in_code = False
    for raw in (text or "").splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            in_code = not in_code
            chunks.append(("class:md-code", line + "\n"))
            continue
        if in_code:
            chunks.append(("class:md-code", line + "\n"))
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            chunks.append(("class:md-h", stripped.lstrip("#").strip() + "\n"))
            continue
        pos = 0
        for match in _BOLD_RE.finditer(line):
            if match.start() > pos:
                chunks.append(("", line[pos : match.start()]))
            chunks.append(("class:md-bold", match.group(1)))
            pos = match.end()
        if pos < len(line):
            chunks.append(("", line[pos:]))
        chunks.append(("", "\n"))
    return chunks


class TuiApp:
    """prompt_toolkit 交互循环。测试不调用 run()。"""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        self._ctx = ctx
        self._config = config or {}
        self.session_id = new_session_id()
        self.agent = DEFAULT_AGENT
        self.lines: list[tuple[str, str]] = []
        self.status = "就绪"
        self.running = False
        self.stream_buf = ""
        self._quit_armed = False
        self._app: Any = None
        self._invalidate: Callable[[], None] | None = None
        self._approval_waiting = False
        self._approval_key: str | None = None
        self._approval_ready = threading.Event()
        self._restore_latest()

    def run(self) -> None:
        if not PROMPT_TOOLKIT_AVAILABLE:
            raise PluginError(_PT_MISSING)
        textarea = TextArea(
            multiline=True,
            prompt="❯ ",
            height=Dimension(min=2, preferred=3),
        )
        header = FormattedTextControl(self._header_frags)
        body = FormattedTextControl(self._body_frags, focusable=False)
        status = FormattedTextControl(lambda: [("class:status", self.status)])
        kb = KeyBindings()

        @kb.add("enter")
        def _send(event: Any) -> None:
            if self._approval_waiting:
                return
            text = textarea.text
            textarea.text = ""
            self.submit(text)
            event.app.invalidate()

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            if self.running:
                self.request_stop()
                self._append("dim", "已请求停止，保留已生成内容")
                event.app.invalidate()
                return
            if self._quit_armed:
                event.app.exit()
                return
            self._quit_armed = True
            self._append("dim", "再按一次 Ctrl+C 退出")
            event.app.invalidate()

        @kb.add("c-d")
        def _eof(event: Any) -> None:
            event.app.exit()

        @kb.add("y")
        @kb.add("n")
        def _approval_yn(event: Any) -> None:
            key = str(event.key_sequence[0].key)
            if self._approval_waiting:
                self._approval_key = key
                self._approval_ready.set()
                return
            event.current_buffer.insert_text(key)

        @kb.add("escape")
        def _approval_esc(event: Any) -> None:
            if self._approval_waiting:
                self._approval_key = "esc"
                self._approval_ready.set()

        layout = Layout(
            HSplit(
                [
                    Window(header, height=1),
                    Window(body),
                    Window(status, height=1),
                    textarea,
                ]
            ),
            focused_element=textarea,
        )
        style = Style.from_dict(
            {
                "title": "bold",
                "user": "bold",
                "dim": "italic ansibrightblack",
                "ok": "ansigreen",
                "err": "ansired",
                "md-h": "bold underline",
                "md-bold": "bold",
                "md-code": "bg:#202020",
                "status": "reverse",
            }
        )
        application = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=True,
        )
        self._app = application
        self._invalidate = application.invalidate
        application.run()

    def submit(self, text: str) -> None:
        self._quit_armed = False
        raw = (text or "").rstrip()
        if not raw:
            return
        cmd = parse_slash(raw)
        if cmd is not None:
            self._handle_slash(cmd)
            return
        if self.running:
            self._append("dim", "上一轮仍在运行，先 /stop 或等待结束")
            return
        self._append("user", raw)
        self.stream_buf = ""
        self.running = True
        self.status = "运行中 · …"
        thread = threading.Thread(target=self._chat_worker, args=(raw,), daemon=True)
        thread.start()

    def request_stop(self) -> None:
        if not self._ctx.has("agent"):
            return
        fn = getattr(self._ctx.require("agent"), "request_stop", None)
        if callable(fn):
            fn()

    def approval_gate(self, payload: dict[str, Any]) -> bool:
        return prompt_approval(
            payload,
            timeout=_APPROVAL_TIMEOUT,
            read_char=self._wait_approval_key,
            printer=lambda line: self._append("dim", line),
        )

    def on_bus(self, event: str, payload: dict[str, Any]) -> None:
        data = payload or {}
        sid = str(data.get("session_id") or "")
        if sid and sid != self.session_id:
            return
        if event in {"step.start", "step.tool_call"} and self.running:
            self.status = running_status(data)
        if event == "answer.delta":
            delta = str(data.get("delta") or "")
            if delta:
                self.stream_buf += delta
                self._replace_stream(self.stream_buf)
        elif event == "answer.done":
            answer = str(data.get("answer") or self.stream_buf)
            self._replace_stream(answer)
            self._append("dim", format_settlement(data))
            self.stream_buf = ""
            self.running = False
            self.status = "就绪"
        else:
            for line in project_event(event, data):
                style = "dim" if event in {"intent.detected", "rag.empty"} else ""
                if event == "step.intercepted" or event == "answer.rejected":
                    style = "err"
                if event == "step.tool_call" and not data.get("error"):
                    style = "ok"
                self._append(style, line)
        self._refresh()

    def _chat_worker(self, text: str) -> None:
        try:
            sessions = self._ctx.require("sessions")
            sessions.chat(self.session_id, text, agent=self.agent)
        except Exception as exc:  # noqa: BLE001 — 投影到界面，不崩主循环
            self._append("err", f"[错误] {exc}")
            self.running = False
            self.status = "就绪"
        self._refresh()

    def _handle_slash(self, cmd: SlashCommand) -> None:
        if cmd.name not in _SLASH_NAMES:
            self._append("dim", unknown_slash_hint(cmd.name))
            return
        if cmd.name == "help":
            self._append("dim", slash_help_text())
            return
        if cmd.name == "quit":
            if self._app is not None:
                self._app.exit()
            return
        if cmd.name == "stop":
            self.request_stop()
            self._append("dim", "已请求停止，保留已生成内容")
            return
        if cmd.name == "new":
            self.session_id = new_session_id()
            self.lines.clear()
            self.stream_buf = ""
            self._append("dim", f"新会话 {self.session_id}")
            return
        if cmd.name == "sessions":
            rows = self._list_sessions()
            for line in format_session_rows(rows):
                self._append("dim", line)
            return
        if cmd.name == "resume":
            rows = self._list_sessions()
            sid = resolve_resume_token(cmd.arg, rows)
            if not sid:
                self._append("err", "无法解析会话，先 /sessions 查看")
                return
            self.session_id = sid
            self._rebuild_from_state()
            return
        if cmd.name == "agent":
            self._handle_agent(cmd.arg)
            return
        if cmd.name == "model":
            self._handle_model(cmd.arg)
            return
        if cmd.name == "export":
            self._handle_export()

    def _handle_agent(self, arg: str) -> None:
        names = self._agent_names()
        if not arg:
            for line in format_agent_rows(names, self.agent):
                self._append("dim", line)
            return
        if arg not in names:
            self._append("err", f"未知 agent：{arg}")
            return
        self.agent = arg
        latest = pick_latest_session(self._list_sessions())
        if latest is not None:
            self.session_id = str(latest["session_id"])
            self._rebuild_from_state()
        else:
            self.session_id = new_session_id()
            self.lines.clear()
        self._append("dim", f"已切换 agent：{arg}")

    def _handle_model(self, arg: str) -> None:
        settings = load_settings()
        cfg = load_providers_config(settings, AgentConfig())
        if not arg:
            for line in format_provider_rows(cfg):
                self._append("dim", line)
            return
        if not arg.isdigit():
            self._append("err", "请输入 /model <序号>")
            return
        updated, message = apply_default_provider(cfg, int(arg))
        if updated is None:
            self._append("err", message)
            return
        data = providers_to_dict(updated)
        settings.update(data)
        save_settings(settings)
        self._append("dim", message)

    def _handle_export(self) -> None:
        sessions = self._ctx.require("sessions")
        state = sessions.get_state(self.session_id, agent=self.agent)
        body = export_markdown(self.session_id, state.title, list(state.turns))
        path = Path(f"{safe_export_name(self.session_id)}.md")
        path.write_text(body, encoding="utf-8")
        self._append("dim", f"已导出 {path}")

    def _restore_latest(self) -> None:
        if not self._ctx.has("sessions"):
            return
        latest = pick_latest_session(self._list_sessions())
        if latest is None:
            self.session_id = new_session_id()
            return
        self.session_id = str(latest["session_id"])
        if latest.get("agent"):
            self.agent = str(latest["agent"])
        self._rebuild_from_state()

    def _rebuild_from_state(self) -> None:
        self.lines.clear()
        self.stream_buf = ""
        sessions = self._ctx.require("sessions")
        state = sessions.get_state(self.session_id, agent=self.agent)
        for turn in state.turns:
            if turn.query_raw:
                self._append("user", turn.query_raw)
            if turn.answer:
                self._append("assistant", turn.answer)
        self._append("dim", f"已恢复 {self.session_id} · {state.title}")

    def _list_sessions(self) -> list[dict[str, Any]]:
        if not self._ctx.has("sessions"):
            return []
        return list(self._ctx.require("sessions").list_sessions(self.agent))

    def _agent_names(self) -> list[str]:
        if not self._ctx.has("agents"):
            return [DEFAULT_AGENT]
        return [item.name for item in self._ctx.require("agents").list()]

    def _header_frags(self) -> StyleAndTextTuples:
        meta = describe_provider(load_providers_config(load_settings(), AgentConfig()))
        provider = meta.get("provider_name") or meta.get("provider_type") or "-"
        model = meta.get("model") or "-"
        title = f"pocket-agent · {provider}·{model} · agent:{self.agent}"
        return [("class:title", title)]

    def _body_frags(self) -> StyleAndTextTuples:
        out: StyleAndTextTuples = []
        for kind, text in self.lines:
            if kind == "user":
                out.append(("class:user", f"❯ {text}\n"))
            elif kind == "assistant":
                out.extend(basic_markdown_ft(text))
                if not text.endswith("\n"):
                    out.append(("", "\n"))
            elif kind == "ok":
                out.append(("class:ok", text + "\n"))
            elif kind == "err":
                out.append(("class:err", text + "\n"))
            else:
                out.append(("class:dim", text + "\n"))
        return out

    def _append(self, kind: str, text: str) -> None:
        self.lines.append((kind, text))

    def _replace_stream(self, text: str) -> None:
        if self.lines and self.lines[-1][0] == "assistant":
            self.lines[-1] = ("assistant", text)
        else:
            self.lines.append(("assistant", text))

    def _wait_approval_key(self, timeout: float) -> str | None:
        self._approval_key = None
        self._approval_ready.clear()
        self._approval_waiting = True
        self._refresh()
        ok = self._approval_ready.wait(timeout=timeout)
        self._approval_waiting = False
        if not ok:
            return None
        return self._approval_key

    def _refresh(self) -> None:
        if self._invalidate is not None:
            self._invalidate()


class UiTuiPlugin:
    """Fast 终端 UI → tui（requires: sessions）。"""

    name = "ui-tui"
    requires: ClassVar[list[str]] = ["sessions"]
    provides: ClassVar[list[str]] = ["tui"]

    def apply(self, ctx: Context, config: dict) -> None:
        if not PROMPT_TOOLKIT_AVAILABLE:
            raise PluginError(_PT_MISSING)
        app = TuiApp(ctx, config)
        if ctx.has("agent"):
            agent = ctx.require("agent")
            if getattr(agent, "approval_gate", None) is None:
                agent.approval_gate = app.approval_gate
        for event in (
            "intent.detected",
            "task.planned",
            "step.start",
            "step.tool_call",
            "step.intercepted",
            "rag.empty",
            "answer.rejected",
            "answer.delta",
            "answer.done",
        ):
            ctx.on(event, lambda payload, name=event: app.on_bus(name, payload))
        ctx.provide("tui", app)
