from __future__ import annotations

from typing import ClassVar

import pytest
from pocket_agent.plugins import builtin_plugins
from pocket_agent.plugins.ui_tui import (
    _PT_MISSING,
    SlashCommand,
    TuiApp,
    UiTuiPlugin,
    _RawModeWithoutVt,
    apply_default_provider,
    bind_tui_keys,
    configure_win32_ime_input,
    cooked_console_mode,
    format_approval_line,
    format_cooked_line,
    format_provider_rows,
    format_settlement,
    needs_cooked_ime_input,
    normalize_console_line,
    parse_slash,
    poll_console_key,
    project_event,
    prompt_approval,
    read_ime_line,
    resolve_approval_key,
    resolve_resume_token,
    running_status,
    slash_help_text,
    strip_vt_input_flag,
    unknown_slash_hint,
)
from pocket_agent.providers import (
    ProviderConfig,
    ProvidersConfig,
    RoutingConfig,
)
from pocket_agent.runtime import Context, PluginError, assemble
from prompt_toolkit.key_binding import KeyBindings


def test_parse_slash_each_command() -> None:
    assert parse_slash("hello") is None
    assert parse_slash("/help") == SlashCommand("help", "")
    assert parse_slash("/") == SlashCommand("help", "")
    assert parse_slash("/new") == SlashCommand("new", "")
    assert parse_slash("/sessions") == SlashCommand("sessions", "")
    assert parse_slash("/resume 2") == SlashCommand("resume", "2")
    assert parse_slash("/resume abc") == SlashCommand("resume", "abc")
    assert parse_slash("/agent") == SlashCommand("agent", "")
    assert parse_slash("/agent 写手") == SlashCommand("agent", "写手")
    assert parse_slash("/model") == SlashCommand("model", "")
    assert parse_slash("/model 1") == SlashCommand("model", "1")
    assert parse_slash("/export") == SlashCommand("export", "")
    assert parse_slash("/stop") == SlashCommand("stop", "")
    assert parse_slash("/quit") == SlashCommand("quit", "")
    assert parse_slash("/unknown") == SlashCommand("unknown", "")
    assert "未知命令 /foo" in unknown_slash_hint("foo")
    assert "/help" in slash_help_text()


def test_resolve_resume_token() -> None:
    rows = [{"session_id": "s-a"}, {"session_id": "s-b"}]
    assert resolve_resume_token("2", rows) == "s-b"
    assert resolve_resume_token("s-a", rows) == "s-a"
    assert resolve_resume_token("9", rows) is None
    assert resolve_resume_token("", rows) is None


def test_project_event_lines() -> None:
    assert project_event(
        "intent.detected", {"mode": "qa", "source": "rule"}
    ) == ["意图：qa（rule）"]
    planned = project_event(
        "task.planned",
        {"goal": "介绍自己", "plan": ["检索", "作答"]},
    )
    assert planned[0] == "任务卡：介绍自己"
    assert "1. 检索" in planned[1]
    assert "2. 作答" in planned[2]
    tool = project_event(
        "step.tool_call",
        {"tool": "write_file", "args": {"path": "a.txt"}, "error": ""},
    )
    assert tool[0].startswith("● 工具 write_file")
    assert tool[0].endswith("✓")
    failed = project_event(
        "step.tool_call",
        {"tool": "write_file", "args": {}, "error": "boom"},
    )
    assert failed[0].endswith("✖")
    assert project_event("step.intercepted", {"reason": "deny"}) == ["已拦截：deny"]
    assert project_event("rag.empty", {}) == ["未检索到资料，由模型自身作答"]
    assert project_event(
        "answer.rejected", {"reasons": ["字数不足", "复读计划"]}
    ) == ["验收未通过：字数不足；复读计划，正在重试"]
    assert project_event("answer.delta", {"delta": "字"}) == []


def test_format_settlement_and_status() -> None:
    line = format_settlement(
        {
            "turn": 3,
            "total_steps": 2,
            "cache_hit_rate": 0.5,
            "intercepts": 1,
            "stop_reason": "qa",
        }
    )
    assert line == "轮次 3 · 2 步 · 缓存命中 50% · 拦截 1 · 停止原因 qa"
    assert running_status({"step_index": 2, "current_tool": "read_file"}) == (
        "运行中 · S2 · read_file…"
    )


def test_needs_cooked_ime_input_for_cursor_not_windows_terminal() -> None:
    assert needs_cooked_ime_input(
        platform="win32", env={"VSCODE_PID": "1"}
    )
    assert needs_cooked_ime_input(
        platform="win32", env={"TERM_PROGRAM": "vscode"}
    )
    assert needs_cooked_ime_input(
        platform="win32", env={"TERM_PROGRAM": "cursor"}
    )
    assert not needs_cooked_ime_input(
        platform="win32", env={"WT_SESSION": "abc"}
    )
    assert not needs_cooked_ime_input(
        platform="linux", env={"VSCODE_PID": "1"}
    )
    assert format_cooked_line("user", "你好") == "❯ 你好"
    assert format_cooked_line("dim", "就绪") == "就绪"
    assert normalize_console_line("你好\r\n") == "你好"
    assert cooked_console_mode(0x0200) & 0x0200 == 0
    assert cooked_console_mode(0) & 0x0002
    written: list[str] = []
    assert read_ime_line("❯ ", writer=written.append, reader=lambda: "你好\r\n") == "你好"
    assert written == ["❯ "]


def test_poll_console_key_yes_and_timeout() -> None:
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    def sleep(delta: float) -> None:
        now["t"] += delta

    hits = {"n": 0}

    def kbhit() -> bool:
        hits["n"] += 1
        return hits["n"] >= 2

    assert poll_console_key(
        1.0,
        clock=clock,
        kbhit=kbhit,
        getwch=lambda: "y",
        sleep=sleep,
    ) == "y"
    now["t"] = 0.0
    assert (
        poll_console_key(
            0.1,
            clock=clock,
            kbhit=lambda: False,
            getwch=lambda: "n",
            sleep=sleep,
        )
        is None
    )


def test_strip_vt_input_flag() -> None:
    assert strip_vt_input_flag(0x0200 | 0x0004) == 0x0004
    assert strip_vt_input_flag(0) == 0


def test_configure_win32_ime_input_replaces_vt_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = {"n": 0}
    created: dict[str, bool] = {}

    class FakeVt:
        def close(self) -> None:
            closed["n"] += 1

    class DummyReader:
        def __init__(self, recognize_paste: bool = True) -> None:
            created["paste"] = recognize_paste
            self.recognize_paste = recognize_paste

    class DummyCM:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    class DummyInp:
        def __init__(self) -> None:
            self.console_input_reader = FakeVt()
            self._use_virtual_terminal_input = True

        def raw_mode(self) -> DummyCM:
            return DummyCM()

    monkeypatch.setattr("pocket_agent.plugins.ui_tui._ConsoleInputReader", DummyReader)
    inp = DummyInp()
    configure_win32_ime_input(inp)
    assert inp._use_virtual_terminal_input is False
    assert isinstance(inp.console_input_reader, DummyReader)
    assert created["paste"] is False
    assert closed["n"] == 1
    assert isinstance(inp.raw_mode(), _RawModeWithoutVt)


def test_approval_letter_bindings_idle_pass_through() -> None:
    waiting = False
    kb = KeyBindings()
    bind_tui_keys(
        kb,
        is_approving=lambda: waiting,
        on_send=lambda _e: None,
        on_newline=lambda _e: None,
        on_ctrl_c=lambda _e: None,
        on_eof=lambda _e: None,
        on_approval_key=lambda _k: None,
    )
    y_binds = [item for item in kb.bindings if item.keys == ("y",)]
    assert y_binds
    assert bool(y_binds[0].filter()) is False
    waiting = True
    assert bool(y_binds[0].filter()) is True


def test_prompt_approval_yes_no_timeout() -> None:
    payload = {"tool": "write_file", "args": {"path": "x"}, "reason": "写文件"}
    printed: list[str] = []
    assert "审批 · write_file" in format_approval_line(payload)
    assert (
        prompt_approval(
            payload,
            read_char=lambda _t: "y",
            printer=printed.append,
        )
        is True
    )
    assert resolve_approval_key("y") is True
    printed.clear()
    assert (
        prompt_approval(
            payload,
            read_char=lambda _t: "n",
            printer=printed.append,
        )
        is False
    )
    printed.clear()
    assert (
        prompt_approval(
            payload,
            read_char=lambda _t: None,
            printer=printed.append,
        )
        is False
    )
    assert "审批超时已拒绝" in printed
    assert resolve_approval_key("esc") is False
    assert resolve_approval_key(None) is False


def test_apply_default_provider_by_index() -> None:
    cfg = ProvidersConfig(
        providers=[
            ProviderConfig(id="a", name="甲", type="llamacpp", base_url="http://x", model="m1"),
            ProviderConfig(id="b", name="乙", type="llamacpp", base_url="http://y", model="m2"),
        ],
        default_provider_id="a",
        routing=RoutingConfig(choose_fill="a", generate="a"),
    )
    listed = format_provider_rows(cfg)
    assert "（默认）" in listed[0]
    updated, message = apply_default_provider(cfg, 2)
    assert updated is not None
    assert updated.default_provider_id == "b"
    assert updated.routing.generate == "b"
    assert "乙" in message
    missing, err = apply_default_provider(cfg, 9)
    assert missing is None
    assert "序号无效" in err


def test_missing_prompt_toolkit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pocket_agent.plugins.ui_tui.PROMPT_TOOLKIT_AVAILABLE", False)
    ctx = Context()
    with pytest.raises(PluginError, match="prompt_toolkit"):
        UiTuiPlugin().apply(ctx, {})
    with pytest.raises(PluginError) as caught:
        UiTuiPlugin().apply(ctx, {})
    assert str(caught.value) == _PT_MISSING


class _FakeLlm:
    def generate(self, prompt: str) -> str:
        return "ok"

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return -1

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return {}


class _FakeLlmAsLlama:
    name = "llm-llamacpp"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm", "llm_local"]

    def apply(self, ctx: Context, config: dict) -> None:
        llm = _FakeLlm()
        ctx.provide("llm_local", llm)
        ctx.provide("llm", llm)


def _tui_yml(profile: str) -> dict:
    return {
        "profile": profile,
        "plugins": [
            {"name": "llm-llamacpp", "config": {"base_url": "http://127.0.0.1:9/v1"}},
            {"name": "tools-core"},
            {"name": "agent-loop", "config": {"max_steps": 2}},
            {"name": "session"},
            {"name": "ui-cli"},
            {"name": "ui-web", "config": {"host": "127.0.0.1", "port": 3998}},
            {"name": "ui-tui"},
        ],
    }


def test_headless_profile_has_no_tui_service() -> None:
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, _tui_yml("headless"))
    try:
        assert "ui-tui" not in ctx.mount_order
        assert "ui-web" not in ctx.mount_order
        assert not ctx.has("tui")
        assert not ctx.has("http")
    finally:
        ctx.unmount_all()


def test_tui_profile_provides_tui_without_web(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pocket_agent.plugins.ui_tui.PROMPT_TOOLKIT_AVAILABLE", True)
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, _tui_yml("tui"))
    try:
        assert "ui-tui" in ctx.mount_order
        assert "ui-web" not in ctx.mount_order
        assert ctx.has("tui")
        assert not ctx.has("http")
        tui = ctx.require("tui")
        assert isinstance(tui, TuiApp)
        assert callable(tui.run)
        gate = ctx.require("agent").approval_gate
        assert callable(gate)
        assert getattr(gate, "__self__", None) is tui
    finally:
        ctx.unmount_all()


def test_builtin_plugins_includes_ui_tui() -> None:
    assert "ui-tui" in builtin_plugins()
