from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any, ClassVar

import httpx
import pytest
from pocket_agent.config import AgentConfig
from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.ui_web import ApprovalBroker, UiWebPlugin, build_app
from pocket_agent.runtime import Context
from pocket_agent.tools import ToolRegistry, tool

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


class _FakeLlm:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)

    def _next(self) -> Any:
        if not self._script:
            return -1
        return self._script.pop(0)

    def generate(self, prompt: str) -> str:
        value = self._next()
        return str(value)

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        value = self._next()
        return int(value)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        value = self._next()
        assert isinstance(value, dict)
        return value


class _FakeLlmPlugin:
    name = "llm-fake"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm"]

    def __init__(self, script: list[Any]) -> None:
        self._script = script

    def apply(self, ctx: Context, config: dict) -> None:
        ctx.provide("llm", _FakeLlm(self._script))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line == "":
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                events.append((event_name, payload))
            event_name = "message"
            data_lines = []
    if data_lines:
        payload = json.loads("\n".join(data_lines))
        events.append((event_name, payload))
    return events


def _mount_stack(ctx: Context, script: list[Any]) -> None:
    ctx.mount(_FakeLlmPlugin(script))
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 8})
    ctx.mount(SessionPlugin())


async def _stream_post(app: Any, path: str, payload: dict) -> tuple[int, str, str]:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", path, json=payload) as response,
    ):
        chunks: list[str] = []
        async for text in response.aiter_text():
            chunks.append(text)
        return (
            response.status_code,
            response.headers.get("content-type", ""),
            "".join(chunks),
        )


async def _get_json(app: Any, path: str) -> Any:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(path)
        assert res.status_code == 200
        return res.json()


def test_sse_chat_event_stream_format() -> None:
    ctx = Context()
    _mount_stack(
        ctx,
        [
            0,
            {"city": "北京"},
            -1,
            "北京天气晴",
        ],
    )
    broker = ApprovalBroker(timeout_sec=2)
    ctx.require("agent").approval_gate = broker.request
    app = build_app(ctx, broker=broker)

    status, ctype, body = asyncio.run(
        _stream_post(app, "/chat", {"session_id": "s1", "message": "查北京天气"})
    )
    assert status == 200
    assert "text/event-stream" in ctype

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert "step.start" in names
    assert "step.tool_call" in names
    assert "answer.done" in names
    answer_ev = next(p for n, p in events if n == "answer.done")
    assert "北京" in answer_ev["answer"]
    assert answer_ev["stop_reason"]
    # 状态条 / 轨迹所需扩展字段
    for key in (
        "llm_ms",
        "tool_ms",
        "cache_hit_rate",
        "real_calls",
        "intercepts",
        "prompt_tokens_max",
        "total_steps",
    ):
        assert key in answer_ev
    start_ev = next(p for n, p in events if n == "step.start")
    assert "step_index" in start_ev
    assert "elapsed_ms" in start_ev
    assert "prompt_tokens_est" in start_ev
    assert "ttft_ms" in start_ev  # 可为 null

    sessions = asyncio.run(_get_json(app, "/sessions"))
    assert any(s["session_id"] == "s1" for s in sessions)
    ctx.unmount_all()


async def _stream_get(app: Any, path: str) -> tuple[int, str, str]:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("GET", path) as response,
    ):
        chunks: list[str] = []
        async for text in response.aiter_text():
            chunks.append(text)
        return (
            response.status_code,
            response.headers.get("content-type", ""),
            "".join(chunks),
        )


def test_sse_get_eventsource_with_overrides() -> None:
    """EventSource 走 GET /chat，overrides 以 query JSON 传递。"""
    from urllib.parse import quote

    ctx = Context()
    _mount_stack(ctx, [-1, "直接答"])
    broker = ApprovalBroker(timeout_sec=2)
    app = build_app(ctx, broker=broker)
    overrides = quote(json.dumps({"max_steps": 3}))
    path = (
        "/chat?message=hello&session_id=es1&run_id=rid-1&overrides=" + overrides
    )
    status, ctype, body = asyncio.run(_stream_get(app, path))
    assert status == 200
    assert "text/event-stream" in ctype
    names = [n for n, _ in _parse_sse(body)]
    assert "answer.done" in names
    ctx.unmount_all()


def test_api_health_reports_llm_flag() -> None:
    ctx = Context()
    _mount_stack(ctx, [-1, "x"])
    broker = ApprovalBroker(timeout_sec=1)
    app = build_app(ctx, broker=broker)
    t0 = time.perf_counter()
    health = asyncio.run(_get_json(app, "/api/health"))
    elapsed = time.perf_counter() - t0
    assert health["ok"] is True
    assert "llm" in health
    assert "base_url" in health
    # HTTP 探测须较快返回，避免拖垮启动页
    assert elapsed < 5.0
    ctx.unmount_all()


def test_sse_cache_hit_and_intercept_markers() -> None:
    ctx = Context()
    _mount_stack(
        ctx,
        [
            0,
            {"city": "北京"},
            0,
            {"city": "北京"},
            -1,
            "已查过",
        ],
    )
    broker = ApprovalBroker(timeout_sec=2)
    ctx.require("agent").approval_gate = broker.request
    app = build_app(ctx, broker=broker)

    _, _, body = asyncio.run(
        _stream_post(app, "/chat", {"message": "查两次北京", "session_id": "cache"})
    )
    events = _parse_sse(body)
    names = [n for n, _ in events]
    assert "step.cache_hit" in names
    assert "step.intercepted" in names
    ctx.unmount_all()


def test_resume_endpoint_streams() -> None:
    ctx = Context()
    _mount_stack(ctx, [-1, "直接回答"])
    broker = ApprovalBroker(timeout_sec=2)
    app = build_app(ctx, broker=broker)
    status, _, body = asyncio.run(
        _stream_post(app, "/resume/run-x", {"task": "无需工具"})
    )
    assert status == 200
    names = [n for n, _ in _parse_sse(body)]
    assert "answer.done" in names
    ctx.unmount_all()


def test_approval_required_timeout_defaults_to_reject() -> None:
    ctx = Context()
    cfg = AgentConfig(max_steps=5)
    registry = ToolRegistry(cfg)

    @tool(registry, "危险写操作", cacheable=False)
    def write_note(text: str) -> str:
        return f"wrote:{text}"

    class _ToolsWrite:
        name = "tools-write"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["tools"]

        def apply(self, c: Context, config: dict) -> None:
            c.provide("tools", registry)

    ctx.mount(_FakeLlmPlugin([0, {"text": "secret"}, -1, "done"]))
    ctx.mount(_ToolsWrite())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 5})
    ctx.mount(SessionPlugin())

    broker = ApprovalBroker(timeout_sec=0.3)
    agent = ctx.require("agent")
    agent.approval_gate = broker.request
    app = build_app(ctx, broker=broker)

    _, _, body = asyncio.run(
        _stream_post(app, "/chat", {"message": "写一条笔记", "session_id": "appr"})
    )
    events = _parse_sse(body)
    names = [n for n, _ in events]
    assert "approval.required" in names
    intercepted = [p for n, p in events if n == "step.intercepted"]
    assert any(p.get("reason") == "approval_denied" for p in intercepted)
    assert not any(
        n == "step.tool_call" and p.get("tool") == "write_note" for n, p in events
    )
    ctx.unmount_all()


def test_approval_approve_allows_write_tool() -> None:
    ctx = Context()
    cfg = AgentConfig(max_steps=5)
    registry = ToolRegistry(cfg)

    @tool(registry, "危险写操作", cacheable=False)
    def write_note(text: str) -> str:
        return f"wrote:{text}"

    class _ToolsWrite:
        name = "tools-write"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["tools"]

        def apply(self, c: Context, config: dict) -> None:
            c.provide("tools", registry)

    ctx.mount(_FakeLlmPlugin([0, {"text": "ok"}, -1, "完成"]))
    ctx.mount(_ToolsWrite())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 5})
    ctx.mount(SessionPlugin())

    broker = ApprovalBroker(timeout_sec=5)
    agent = ctx.require("agent")
    agent.approval_gate = broker.request
    app = build_app(ctx, broker=broker)

    def _auto_approve() -> None:
        deadline = time.time() + 3
        while time.time() < deadline:
            with broker._lock:
                ids = list(broker._events)
            if ids:
                broker.resolve(ids[0], True)
                return
            time.sleep(0.02)

    helper = threading.Thread(target=_auto_approve, daemon=True)
    helper.start()
    _, _, body = asyncio.run(
        _stream_post(app, "/chat", {"message": "写笔记", "session_id": "appr2"})
    )
    helper.join(timeout=2)

    events = _parse_sse(body)
    assert any(n == "step.tool_call" and p.get("tool") == "write_note" for n, p in events)
    ctx.unmount_all()


def test_unmount_releases_http_port() -> None:
    port = _free_port()
    ctx = Context()
    _mount_stack(ctx, [-1, "hi"])
    ctx.mount(UiWebPlugin(), {"host": "127.0.0.1", "port": port})
    assert ctx.has("http")
    assert _port_open("127.0.0.1", port)

    ctx.unmount("ui-web")
    deadline = time.time() + 5
    while _port_open("127.0.0.1", port) and time.time() < deadline:
        time.sleep(0.05)
    assert not _port_open("127.0.0.1", port)
    ctx.unmount_all()


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0
