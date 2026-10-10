from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from pocket_agent.config import (
    AgentConfig,
    extract_base_from_pocket,
    load_settings,
    redact_secrets,
    save_settings,
)
from pocket_agent.config_schema import get_config_schema
from pocket_agent.events import EventStore
from pocket_agent.runtime import Context, PluginError, PluginInUse
from pydantic import BaseModel, Field

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
    overrides: dict[str, Any] = Field(default_factory=dict)


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


def attach_event_persistence(
    ctx: Context,
    store: EventStore,
) -> list[Callable[[], None]]:
    """订阅总线事件并同步落盘；返回 disposer 列表。"""

    def _persist(event_name: str) -> Callable[[dict[str, Any]], None]:
        def _handler(payload: dict[str, Any]) -> None:
            sid = str(payload.get("session_id") or "default")
            store.append(sid, event_name, dict(payload))

        return _handler

    return [ctx.on(name, _persist(name)) for name in _BUS_EVENTS]


def build_app(
    ctx: Context,
    *,
    broker: ApprovalBroker,
    event_store: EventStore | None = None,
    settings_path: Path | str | None = None,
    pocket_data: dict[str, Any] | None = None,
    plugin_catalog: Mapping[str, Any] | None = None,
    plugin_configs: dict[str, dict[str, Any]] | None = None,
    active_runs: dict[str, Any] | None = None,
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
    config_lock = threading.Lock()
    store = event_store or EventStore()
    settings_file = Path(settings_path) if settings_path is not None else Path("settings.json")
    pocket = pocket_data or {}
    catalog = plugin_catalog or {}
    configs = plugin_configs or {}
    runs = active_runs if active_runs is not None else {}

    assets_dir = _STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    def _base_layer() -> dict[str, Any]:
        return extract_base_from_pocket(pocket)

    def _resolved_config(overrides: dict[str, Any] | None = None) -> AgentConfig:
        return AgentConfig.from_layers(
            base=_base_layer(),
            settings=load_settings(settings_file),
            overrides=overrides,
        )

    def _apply_live_config(config: AgentConfig) -> None:
        if ctx.has("agent"):
            ctx.require("agent").apply_config(config)

    def _probe_llm(base_url: str, timeout: float = 1.5) -> tuple[bool, str]:
        """探测 llama-server / OpenAI 兼容端点是否可达。"""
        import urllib.error
        import urllib.request

        root = base_url.rstrip("/")
        candidates = [f"{root}/health", f"{root}/v1/models", f"{root}/models"]
        last = "unreachable"
        for url in candidates:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if 200 <= int(resp.status) < 500:
                        return True, ""
            except urllib.error.HTTPError as exc:
                # 401/404 仍说明服务在线
                if exc.code in {401, 403, 404}:
                    return True, ""
                last = str(exc)
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
        return False, last

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse("<h1>pocket-agent</h1><p>static UI missing</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        cfg = _resolved_config()
        ok, message = _probe_llm(cfg.base_url)
        return {
            "ok": True,
            "llm": ok,
            "base_url": cfg.base_url,
            "message": message,
            "hint": (
                None
                if ok
                else "启动 llama-server 后点击重试；示例：./llama-server -m model.gguf --port 8080"
            ),
        }

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

    # ── 配置 API ──────────────────────────────────────────

    @app.get("/api/config/schema")
    def api_config_schema() -> list[dict[str, Any]]:
        return get_config_schema()

    @app.get("/api/config")
    def api_config_get() -> dict[str, Any]:
        cfg = _resolved_config()
        return redact_secrets(cfg.to_dict())

    @app.put("/api/config")
    def api_config_put(body: dict[str, Any]) -> JSONResponse:
        with config_lock:
            current = load_settings(settings_file)
            known = AgentConfig.known_keys()
            for key, value in body.items():
                if key in known:
                    # 脱敏回传值不覆盖明文密钥
                    if key == "api_key" and isinstance(value, str) and "***" in value:
                        continue
                    current[key] = value
            save_settings(current, settings_file)
            merged = _resolved_config()
            _apply_live_config(merged)
        return JSONResponse({"ok": True, "config": redact_secrets(merged.to_dict())})

    # ── 插件 API ──────────────────────────────────────────

    @app.get("/api/plugins")
    def api_plugins() -> list[dict[str, Any]]:
        mounted = set(ctx.mount_order)
        rows: list[dict[str, Any]] = []
        for name, plugin in catalog.items():
            rows.append(
                {
                    "name": name,
                    "requires": list(getattr(plugin, "requires", []) or []),
                    "provides": list(getattr(plugin, "provides", []) or []),
                    "enabled": name in mounted,
                    "active": name in mounted,
                }
            )
        return rows

    @app.post("/api/plugins/{name}/enable")
    def api_plugin_enable(name: str) -> JSONResponse:
        if name not in catalog:
            return JSONResponse({"ok": False, "error": f"unknown plugin: {name}"}, 404)
        if name in ctx.mount_order:
            return JSONResponse({"ok": True, "status": "already_enabled"})
        plugin = catalog[name]
        cfg = dict(configs.get(name) or {})
        prepare = getattr(plugin, "prepare", None)
        if callable(prepare):
            prepare(cfg)
        try:
            ctx.mount(plugin, cfg)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        return JSONResponse({"ok": True, "status": "enabled"})

    @app.post("/api/plugins/{name}/disable")
    def api_plugin_disable(name: str) -> JSONResponse:
        if name == "ui-web":
            return JSONResponse(
                {"ok": False, "error": "cannot disable ui-web while serving"},
                400,
            )
        if name not in ctx.mount_order:
            return JSONResponse({"ok": True, "status": "already_disabled"})
        try:
            ctx.unmount(name, cascade=False)
        except PluginInUse as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 409)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        return JSONResponse({"ok": True, "status": "disabled"})

    # ── 会话事件 / 导出 ───────────────────────────────────

    @app.get("/api/sessions/{session_id}/events")
    def api_session_events(
        session_id: str,
        before: int | None = Query(None),
        limit: int = Query(50, ge=0, le=500),
    ) -> dict[str, Any]:
        return store.list_before(session_id, before=before, limit=limit)

    @app.get("/api/sessions/{session_id}/export")
    def api_session_export(session_id: str) -> dict[str, Any]:
        events = store.export_all(session_id)
        turns: list[dict[str, Any]] = []
        if ctx.has("sessions"):
            try:
                state = ctx.require("sessions").get_state(session_id)
                turns = [
                    {
                        "turn": t.turn,
                        "query_raw": t.query_raw,
                        "query_rewritten": t.query_rewritten,
                        "answer": t.answer,
                    }
                    for t in state.turns
                ]
            except Exception:  # noqa: BLE001
                turns = []
        return {
            "session_id": session_id,
            "turns": turns,
            "events": events,
            "warning": "日志可能包含路径、prompt 与工具参数，分享前请脱敏",
        }

    # ── 停止 ──────────────────────────────────────────────

    @app.post("/api/chat/{run_id}/stop")
    def api_chat_stop(run_id: str) -> JSONResponse:
        runner = runs.get(run_id)
        if runner is None:
            return JSONResponse({"ok": False, "error": "run not found"}, 404)
        runner.request_stop()
        return JSONResponse({"ok": True, "run_id": run_id})

    def _sse_stream(
        run_fn: Callable[[], None],
    ) -> Iterator[bytes]:
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
                q.put({"event": "error", "data": {"message": str(exc)}})
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

    def _run_chat(
        *,
        message: str,
        session_id: str,
        run_id: str | None,
        overrides: dict[str, Any] | None,
    ) -> None:
        agent = ctx.require("agent")
        rid = run_id or str(uuid.uuid4())
        base_cfg = _resolved_config()
        live = AgentConfig.from_layers(
            base=_base_layer(),
            settings=load_settings(settings_file),
            overrides=overrides or {},
        )
        agent.apply_config(live)
        runs[rid] = agent
        try:
            if ctx.has("sessions") and run_id is None:
                # 会话路径：自动 run_id 仅用于 stop，不启用 checkpoint resume
                ctx.require("sessions").chat(session_id, message)
            else:
                agent.run(message, run_id=run_id, session_id=session_id)
        finally:
            runs.pop(rid, None)
            # overrides 不污染 settings：恢复 settings 层
            agent.apply_config(base_cfg)

    @app.post("/chat")
    @app.post("/api/chat")
    def chat_post(body: ChatIn) -> StreamingResponse:
        message = str(body.message or body.task or "").strip()
        if not message:
            return StreamingResponse(
                iter([b'event: error\ndata: {"message":"empty message"}\n\n']),
                media_type="text/event-stream",
            )
        session_id = str(body.session_id or "default")
        run_id_s = str(body.run_id) if body.run_id else None
        overrides = dict(body.overrides or {})

        def _run() -> None:
            _run_chat(
                message=message,
                session_id=session_id,
                run_id=run_id_s,
                overrides=overrides,
            )

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
        overrides: str | None = Query(None),
    ) -> StreamingResponse:
        """EventSource 友好的 GET SSE。"""
        text = message.strip()
        if not text:
            return StreamingResponse(
                iter([b'event: error\ndata: {"message":"empty message"}\n\n']),
                media_type="text/event-stream",
            )
        parsed: dict[str, Any] = {}
        if overrides:
            try:
                raw = json.loads(overrides)
                if isinstance(raw, dict):
                    parsed = {k: v for k, v in raw.items() if not str(k).startswith("_")}
            except json.JSONDecodeError:
                parsed = {}

        def _run() -> None:
            _run_chat(
                message=text,
                session_id=session_id,
                run_id=run_id,
                overrides=parsed or None,
            )

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
            _run_chat(
                message=task,
                session_id="default",
                run_id=run_id,
                overrides=None,
            )

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
        settings_path = Path(str(config.get("settings_path") or "settings.json"))
        events_root = config.get("events_root")
        event_store = EventStore(
            Path(events_root) if events_root is not None else None
        )
        broker = ApprovalBroker(timeout_sec=timeout)

        # pocket.yml 默认层（可由装配方注入）
        pocket_data = config.get("pocket_data")
        if not isinstance(pocket_data, dict):
            pocket_path = config.get("pocket_path")
            if pocket_path:
                from pocket_agent.runtime import load_pocket_config

                pocket_data = load_pocket_config(pocket_path)
            else:
                pocket_data = {}

        plugin_catalog = config.get("plugin_catalog") or {}
        plugin_configs = config.get("plugin_configs") or {}
        if not isinstance(plugin_catalog, dict):
            plugin_catalog = {}
        if not isinstance(plugin_configs, dict):
            plugin_configs = {}

        active_runs: dict[str, Any] = {}

        agent = ctx.require("agent")
        agent.approval_gate = broker.request
        # 启动时应用 settings 层
        agent.apply_config(
            AgentConfig.from_layers(
                base=extract_base_from_pocket(pocket_data),
                settings=load_settings(settings_path),
            )
        )

        # 事件持久化：总线全部步骤事件同步落盘
        persist_disposers = attach_event_persistence(ctx, event_store)

        app = build_app(
            ctx,
            broker=broker,
            event_store=event_store,
            settings_path=settings_path,
            pocket_data=pocket_data,
            plugin_catalog=plugin_catalog,
            plugin_configs=plugin_configs,
            active_runs=active_runs,
        )
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
            for d in persist_disposers:
                d()
            server.should_exit = True
            if hasattr(server, "force_exit"):
                server.force_exit = True
            thread.join(timeout=8)
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
                "event_store": event_store,
                "settings_path": settings_path,
                "active_runs": active_runs,
            },
        )
        logger.info("ui-web listening on http://%s:%s", host, port)
