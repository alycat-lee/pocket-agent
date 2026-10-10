from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from pocket_agent.runtime import Context, PluginError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_BUS_EVENTS = (
    "step.start",
    "step.tool_call",
    "step.cache_hit",
    "step.intercepted",
    "answer.done",
    "approval.required",
)
_DEFAULT_APPROVAL_TIMEOUT = 60.0


class ChatIn(BaseModel):
    message: str = ""
    task: str = ""
    session_id: str = "default"
    run_id: str | None = None


class ResumeIn(BaseModel):
    message: str = ""
    task: str = ""


class ApproveIn(BaseModel):
    approval_id: str = ""
    approved: bool = False


class ApprovalBroker:
    """跨线程审批：emit approval.required 后阻塞等待 Web 确认。"""

    def __init__(self, timeout_sec: float = _DEFAULT_APPROVAL_TIMEOUT) -> None:
        self._timeout = timeout_sec
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._results: dict[str, bool] = {}

    def request(self, payload: dict[str, Any]) -> bool:
        approval_id = str(payload.get("approval_id") or "")
        if not approval_id:
            return False
        ev = threading.Event()
        with self._lock:
            self._events[approval_id] = ev
        timed_out = not ev.wait(self._timeout)
        with self._lock:
            self._events.pop(approval_id, None)
            approved = self._results.pop(approval_id, False)
        if timed_out:
            return False
        return approved

    def resolve(self, approval_id: str, approved: bool) -> bool:
        with self._lock:
            if approval_id not in self._events:
                return False
            self._results[approval_id] = approved
            self._events[approval_id].set()
        return True


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def build_app(
    ctx: Context,
    *,
    broker: ApprovalBroker,
) -> Any:
    """构造 FastAPI app（供插件启动与 ASGI 测试复用）。"""
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise PluginError(
            "ui-web requires fastapi/uvicorn: pip install pocket-agent[web]"
        ) from exc

    app = FastAPI(title="pocket-agent web")
    chat_lock = threading.Lock()
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse("<h1>pocket-agent</h1><p>static UI missing</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/sessions")
    def sessions() -> list[dict[str, Any]]:
        if not ctx.has("sessions"):
            return []
        return list(ctx.require("sessions").list_sessions())

    @app.post("/approve")
    def approve(body: ApproveIn) -> JSONResponse:
        ok = broker.resolve(str(body.approval_id), bool(body.approved))
        return JSONResponse(
            {
                "ok": ok,
                "approval_id": body.approval_id,
                "approved": body.approved,
            }
        )

    def _sse_stream(
        run_fn: Callable[[], None],
    ) -> Iterator[bytes]:
        # 单飞：避免多路 SSE 交叉订阅同一事件总线
        chat_lock.acquire()
        q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        disposers: list[Callable[[], None]] = []
        released = False

        def _release_lock() -> None:
            nonlocal released
            if not released:
                released = True
                chat_lock.release()

        def _make_handler(event_name: str) -> Callable[[dict[str, Any]], None]:
            def _handler(payload: dict[str, Any]) -> None:
                q.put({"event": event_name, "data": payload})

            return _handler

        for name in _BUS_EVENTS:
            disposers.append(ctx.on(name, _make_handler(name)))

        def _worker() -> None:
            try:
                run_fn()
            except Exception as exc:  # noqa: BLE001 — 推送到 SSE 后结束
                q.put(
                    {
                        "event": "error",
                        "data": {"message": str(exc)},
                    }
                )
            finally:
                q.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                event = str(item["event"])
                data = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {event}\ndata: {data}\n\n".encode()
        finally:
            for dispose in disposers:
                dispose()
            thread.join(timeout=30)
            _release_lock()

    @app.post("/chat")
    def chat_post(body: ChatIn) -> StreamingResponse:
        message = str(body.message or body.task or "").strip()
        if not message:
            return StreamingResponse(
                iter([b'event: error\ndata: {"message":"empty message"}\n\n']),
                media_type="text/event-stream",
            )
        session_id = str(body.session_id or "default")
        run_id_s = str(body.run_id) if body.run_id else None

        def _run() -> None:
            if ctx.has("sessions") and run_id_s is None:
                ctx.require("sessions").chat(session_id, message)
            else:
                ctx.require("agent").run(message, run_id=run_id_s)

        return StreamingResponse(
            _sse_stream(_run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/chat")
    def chat_get(
        message: str = Query(...),
        session_id: str = Query("default"),
        run_id: str | None = Query(None),
    ) -> StreamingResponse:
        """EventSource 友好的 GET SSE。"""
        text = message.strip()
        if not text:
            return StreamingResponse(
                iter([b'event: error\ndata: {"message":"empty message"}\n\n']),
                media_type="text/event-stream",
            )

        def _run() -> None:
            if ctx.has("sessions") and not run_id:
                ctx.require("sessions").chat(session_id, text)
            else:
                ctx.require("agent").run(text, run_id=run_id)

        return StreamingResponse(
            _sse_stream(_run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/resume/{run_id}")
    def resume(run_id: str, body: ResumeIn) -> StreamingResponse:
        task = str(body.message or body.task or "").strip()
        if not task:
            return StreamingResponse(
                iter([b'event: error\ndata: {"message":"empty task"}\n\n']),
                media_type="text/event-stream",
            )

        def _run() -> None:
            ctx.require("agent").run(task, run_id=run_id)

        return StreamingResponse(
            _sse_stream(_run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


class UiWebPlugin:
    """FastAPI + SSE Web UI → http（requires: agent, sessions）。"""

    name = "ui-web"
    requires: ClassVar[list[str]] = ["agent", "sessions"]
    provides: ClassVar[list[str]] = ["http"]

    def apply(self, ctx: Context, config: dict) -> None:
        try:
            import uvicorn
        except ImportError as exc:
            raise PluginError(
                "ui-web requires fastapi/uvicorn: pip install pocket-agent[web]"
            ) from exc

        host = str(config.get("host") or "127.0.0.1")
        port = int(config.get("port") or 3000)
        timeout = float(config.get("approval_timeout") or _DEFAULT_APPROVAL_TIMEOUT)
        broker = ApprovalBroker(timeout_sec=timeout)

        agent = ctx.require("agent")
        agent.approval_gate = broker.request

        app = build_app(ctx, broker=broker)
        uvi_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=str(config.get("log_level") or "warning"),
            access_log=False,
        )
        server = uvicorn.Server(uvi_config)
        thread = threading.Thread(target=server.run, name="ui-web-uvicorn", daemon=True)
        thread.start()

        deadline = time.time() + 10.0
        while not server.started and thread.is_alive() and time.time() < deadline:
            time.sleep(0.02)
        if not server.started:
            server.should_exit = True
            thread.join(timeout=2)
            raise PluginError(f"ui-web failed to bind {host}:{port}")

        closed = threading.Event()

        def dispose() -> None:
            if closed.is_set():
                return
            closed.set()
            agent.approval_gate = None
            server.should_exit = True
            # uvicorn 0.2x: force exit flag
            if hasattr(server, "force_exit"):
                server.force_exit = True
            thread.join(timeout=8)
            # 等端口释放（计入 disposer 验收）
            release_deadline = time.time() + 5.0
            while _port_in_use(host, port) and time.time() < release_deadline:
                time.sleep(0.05)
            if _port_in_use(host, port):
                logger.warning("ui-web port %s:%s still in use after unmount", host, port)

        ctx.on_dispose(dispose)
        ctx.provide(
            "http",
            {
                "host": host,
                "port": port,
                "app": app,
                "server": server,
                "broker": broker,
            },
        )
        logger.info("ui-web listening on http://%s:%s", host, port)
