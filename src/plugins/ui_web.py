from __future__ import annotations

import base64
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
from urllib.parse import urlparse

import httpx
from pocket_agent.agents import AgentNameError
from pocket_agent.config import (
    AgentConfig,
    extract_base_from_pocket,
    load_settings,
    redact_secrets,
    save_settings,
)
from pocket_agent.config_schema import get_config_schema
from pocket_agent.events import EventStore
from pocket_agent.llm_client import resolve_trust_env
from pocket_agent.prompts import builtin_entries
from pocket_agent.runtime import Context, PluginError, PluginInUse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _read_build_info() -> dict[str, str] | None:
    """读取构建产物中的 build-info.json；不存在时返回 None。"""
    info_path = _STATIC_DIR / "build-info.json"
    if not info_path.is_file():
        return None
    try:
        import json as _json

        return _json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


_BUILD_INFO = _read_build_info()
_BUS_EVENTS = (
    "step.start",
    "step.tool_call",
    "step.cache_hit",
    "step.intercepted",
    "answer.done",
    "approval.required",
    "session.renamed",
    "intent.detected",
    "session.permission_changed",
    "session.workspace_changed",
    "step.thinking",
    "task.planned",
    "task.step_done",
    "answer.rejected",
    "run.start",
    "orchestrator.start",
    "orchestrator.done",
    "worker.start",
    "worker.done",
    "merger.start",
    "merger.done",
    "rag.empty",
    "answer.delta",
    "llm.call",
)
_PERSIST_EVENTS = tuple(name for name in _BUS_EVENTS if name != "answer.delta")
_DEFAULT_APPROVAL_TIMEOUT = 60.0


def join_worker_thread(thread: threading.Thread, timeout: float = 30.0) -> None:
    """Join a worker unless cleanup is already running on that worker.

    StreamingResponse 生成器被 GC / close 时，可能在工作线程上执行 finally；
    此时 join 自身会抛 RuntimeError: cannot join current thread。
    """
    if thread is threading.current_thread():
        return
    thread.join(timeout=timeout)


def probe_llm_full(
    base_url: str,
    *,
    timeout: float = 2.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """HTTP 探测 LLM 端点；返回 {ok, base_url, latency_ms, error}。

    本地地址强制 trust_env=False，避免系统代理劫持 localhost。
    """
    t0 = time.perf_counter()
    root = base_url.rstrip("/") if base_url else ""
    if not root:
        return {
            "ok": False,
            "base_url": base_url,
            "latency_ms": 0,
            "error": "base_url 为空",
        }
    parsed = urlparse(root if "://" in root else f"http://{root}")
    origin = f"{parsed.scheme or 'http'}://{parsed.hostname or '127.0.0.1'}"
    if parsed.port:
        origin = f"{origin}:{parsed.port}"
    candidates = [f"{root}/models", f"{origin}/v1/models", f"{origin}/health"]
    urls: list[str] = []
    for item in candidates:
        if item not in urls:
            urls.append(item)
    trust_env = resolve_trust_env(root, None)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_err = "unreachable"
    for url in urls:
        try:
            with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
                resp = client.get(url, headers=headers)
            latency = int((time.perf_counter() - t0) * 1000)
            if 200 <= resp.status_code < 500:
                return {
                    "ok": True,
                    "base_url": base_url,
                    "latency_ms": latency,
                    "error": "",
                }
            last_err = f"HTTP {resp.status_code}"
        except httpx.ConnectError as exc:
            if "refused" in str(exc).lower() or "[errno 111]" in str(exc).lower():
                last_err = "连接被拒绝：llama-server 未启动或端口不对"
            else:
                last_err = f"连接失败：{exc}"
        except httpx.TimeoutException:
            last_err = "连接超时：服务响应过慢或防火墙拦截"
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                last_err = (
                    f"HTTP {exc.response.status_code}：推理服务异常，"
                    "请重启 llama-server"
                )
            else:
                last_err = f"HTTP {exc.response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
    return {
        "ok": False,
        "base_url": base_url,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "error": last_err,
    }


def probe_provider(
    base_url: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """测试 provider：先 GET /models 测连通，再发 max_tokens=1 测生成。"""
    root = base_url.rstrip("/") if base_url else ""
    if not root:
        return {
            "ok": False,
            "connect_ms": 0,
            "generate_ms": 0,
            "model_list": [],
            "error": "base_url 为空",
        }
    trust_env = resolve_trust_env(root, None)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"
    # 1) 连通：GET /models（或 /v1/models）
    models_url = (
        f"{root}/models"
        if root.endswith("/v1")
        else f"{root}/v1/models"
    )
    connect_ms = 0
    model_list: list[str] = []
    try:
        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
            resp = client.get(models_url, headers=headers)
        connect_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code >= 400:
            return {
                "ok": False,
                "connect_ms": connect_ms,
                "generate_ms": 0,
                "model_list": [],
                "error": f"连通失败：HTTP {resp.status_code} {resp.text[:200]}",
            }
        data = resp.json()
        items = data.get("data") or data.get("models") or []
        model_list = [
            str(m.get("id") or m.get("model") or m.get("name") or "")
            for m in items
            if isinstance(m, dict)
        ]
        model_list = [m for m in model_list if m]
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "connect_ms": connect_ms,
            "generate_ms": 0,
            "model_list": [],
            "error": f"连通失败：{exc}",
        }
    # 2) 生成：POST /chat/completions max_tokens=1
    chat_url = (
        f"{root}/chat/completions"
        if root.endswith("/v1")
        else f"{root}/v1/chat/completions"
    )
    use_model = model or (model_list[0] if model_list else "local-model")
    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    generate_ms = 0
    try:
        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
            resp = client.post(chat_url, json=payload, headers=headers)
        generate_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code >= 400:
            return {
                "ok": False,
                "connect_ms": connect_ms,
                "generate_ms": generate_ms,
                "model_list": model_list,
                "error": (
                    f"生成失败：HTTP {resp.status_code} {resp.text[:200]}"
                ),
            }
        return {
            "ok": True,
            "connect_ms": connect_ms,
            "generate_ms": generate_ms,
            "model_list": model_list,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "connect_ms": connect_ms,
            "generate_ms": generate_ms,
            "model_list": model_list,
            "error": f"生成失败：{exc}",
        }


class ChatIn(BaseModel):
    message: str = ""
    task: str = ""
    session_id: str = "default"
    run_id: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None
    permission: str = "workspace"
    agent: str = "default"


class ResumeIn(BaseModel):
    message: str = ""
    task: str = ""


class ApproveIn(BaseModel):
    approval_id: str = ""
    approved: bool = False


class ProviderTestIn(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str | None = None
    timeout: float = 10.0


class ProviderIn(BaseModel):
    name: str
    type: str = "llamacpp"
    base_url: str
    api_key: str = ""
    model: str = ""
    sampling: dict[str, Any] | None = None
    thinking: str | None = None
    trust_env: bool | None = None


class ProviderUpdateIn(BaseModel):
    name: str | None = None
    type: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    sampling: dict[str, Any] | None = None
    thinking: str | None = None
    trust_env: bool | None = None


class KnowledgeIngestIn(BaseModel):
    name: str = "note.txt"
    text: str = ""
    content: str = ""
    content_b64: str = ""


class KnowledgeSearchIn(BaseModel):
    query: str = ""
    top_k: int | None = None


class RoutingIn(BaseModel):
    choose_fill: str
    generate: str


class AgentIn(BaseModel):
    name: str = ""
    description: str = ""
    provider_id: str | None = None
    tools: list[str] | None = None
    permission: str = "workspace"
    prompt: str = ""


class AgentUpdateIn(BaseModel):
    description: str | None = None
    provider_id: str | None = None
    tools: list[str] | None = None
    permission: str | None = None
    prompt: str | None = None


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


def sse_accepts_session(session_id: str, payload: Mapping[str, Any]) -> bool:
    """本会话事件入队；无 session_id 的全局/运行级事件（如 error）仍放行。"""
    sid = str(payload.get("session_id") or "")
    return (not sid) or sid == session_id


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

    return [ctx.on(name, _persist(name)) for name in _PERSIST_EVENTS]


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
    routing_from_settings = settings_path is not None
    settings_file = Path(settings_path) if routing_from_settings else Path("settings.json")
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
        if ctx.has("orchestrator"):
            ctx.require("orchestrator").apply_config(config)
        if ctx.has("agents"):
            ctx.require("agents").set_root(config.agents_root)

    def _providers_settings() -> dict[str, Any]:
        return load_settings(settings_file)

    def _active_provider_endpoint() -> tuple[str, str | None]:
        """health / 探测走当前 generate 路由（无 providers 则用扁平 base_url）。"""
        settings = _providers_settings()
        raw = settings.get("providers")
        ac = _resolved_config()
        if isinstance(raw, list) and raw:
            from pocket_agent.providers import load_providers_config

            pcfg = load_providers_config(settings, ac)
            pid = pcfg.routing.generate or pcfg.default_provider_id
            pc = next((p for p in pcfg.providers if p.id == pid), None)
            if pc is None and pcfg.providers:
                pc = pcfg.providers[0]
            if pc is not None:
                return pc.base_url, pc.api_key or None
        return ac.base_url, ac.api_key or None

    def _run_provider_meta(config: AgentConfig) -> dict[str, str]:
        settings = _providers_settings()
        raw = settings.get("providers")
        if isinstance(raw, list) and raw:
            from pocket_agent.providers import describe_provider, load_providers_config

            pcfg = load_providers_config(settings, config)
            return describe_provider(pcfg, role="generate", fallback_model=config.model)
        return {
            "provider_name": "本地模型",
            "provider_type": "llamacpp",
            "model": config.model,
        }

    def _sync_prompt_variant(config: AgentConfig | None = None) -> None:
        if not ctx.has("prompt_registry"):
            return
        live = config if config is not None else _resolved_config()
        meta = _run_provider_meta(live)
        ctx.require("prompt_registry").select_for_provider(meta["provider_type"])
        if ctx.has("orchestrator"):
            ctx.require("orchestrator").provider_type = meta["provider_type"]

    def _bind_routed_llm(config: AgentConfig) -> list[tuple[Any, Any]]:
        if not routing_from_settings:
            return []
        settings = _providers_settings()
        raw = settings.get("providers")
        if not isinstance(raw, list) or not raw:
            return []
        from pocket_agent.providers import ProviderRegistry, load_providers_config

        routed = ProviderRegistry(load_providers_config(settings, config), config)
        saved: list[tuple[Any, Any]] = []
        for key in ("agent", "sessions", "writer"):
            if ctx.has(key):
                svc = ctx.require(key)
                if hasattr(svc, "_llm"):
                    saved.append((svc, svc._llm))
                    svc._llm = routed
        return saved

    def _probe_llm(base_url: str, timeout: float = 2.0) -> tuple[bool, str]:
        """HTTP 探测 LLM 是否可用；返回 (ok, 可操作错误文案)。"""
        result = probe_llm_full(base_url, timeout=timeout)
        return result["ok"], result["error"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse("<h1>pocket-agent</h1><p>static UI missing</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        base_url, api_key = _active_provider_endpoint()
        result = probe_llm_full(base_url, timeout=2.0, api_key=api_key)
        ok = bool(result["ok"])
        message = str(result.get("error") or "")
        return {
            "ok": True,
            "llm": ok,
            "base_url": base_url,
            "message": message,
            "ui_build": _BUILD_INFO,
            "hint": (
                None
                if ok
                else "模型未连接。请启动本地 llama-server，或在设置页添加可用的 OpenAI 兼容端点后重试。"
            ),
        }

    @app.get("/api/health/llm")
    def api_health_llm() -> dict[str, Any]:
        base_url, api_key = _active_provider_endpoint()
        return probe_llm_full(base_url, timeout=2.0, api_key=api_key)

    @app.post("/api/providers/test")
    def api_providers_test(body: ProviderTestIn) -> dict[str, Any]:
        return probe_provider(
            body.base_url,
            api_key=body.api_key or None,
            model=body.model or None,
            timeout=body.timeout or 10.0,
        )

    @app.get("/sessions")
    def sessions(agent: str | None = Query(None)) -> list[dict[str, Any]]:
        if not ctx.has("sessions"):
            return []
        return list(ctx.require("sessions").list_sessions(agent=agent))

    @app.delete("/api/sessions/{session_id}")
    def api_session_delete(
        session_id: str,
        agent: str | None = Query(None),
    ) -> JSONResponse:
        if not ctx.has("sessions"):
            return JSONResponse({"ok": False, "error": "sessions not available"}, 400)
        ctx.require("sessions").delete_session(session_id, agent=agent)
        return JSONResponse({"ok": True, "session_id": session_id})

    @app.get("/api/sessions/{session_id}")
    def api_session_detail(
        session_id: str,
        agent: str | None = Query(None),
    ) -> JSONResponse:
        if not ctx.has("sessions"):
            return JSONResponse({"ok": False, "error": "sessions not available"}, 400)
        sessions = ctx.require("sessions")
        rows = sessions.list_sessions(agent=agent)
        match = next((row for row in rows if row.get("session_id") == session_id), None)
        if match is None:
            return JSONResponse({"ok": False, "error": "session not found"}, 404)
        owner = agent or match.get("agent")
        try:
            state = sessions.get_state(session_id, agent=owner or None)
        except KeyError:
            return JSONResponse({"ok": False, "error": "session not found"}, 404)
        return JSONResponse(
            {
                "ok": True,
                "session_id": state.session_id,
                "title": state.title,
                "agent": state.agent,
                "turns": [
                    {
                        "turn": t.turn,
                        "query_raw": t.query_raw,
                        "answer": t.answer,
                    }
                    for t in state.turns
                ],
            }
        )

    def _agent_payload(profile: Any) -> dict[str, Any]:
        svc = ctx.require("agents")
        return {
            **profile.to_dict(),
            "prompt": svc.read_prompt(profile.name),
            "root": str(svc.agent_dir(profile.name)),
        }

    @app.get("/api/agents")
    def api_agents_list() -> dict[str, Any]:
        if not ctx.has("agents"):
            return {"agents": [], "root": ""}
        svc = ctx.require("agents")
        return {
            "root": str(svc.root()),
            "agents": [_agent_payload(item) for item in svc.list()],
        }

    @app.post("/api/agents")
    def api_agents_create(body: AgentIn) -> JSONResponse:
        if not ctx.has("agents"):
            return JSONResponse({"ok": False, "error": "未加载 agents 插件"}, 503)
        svc = ctx.require("agents")
        try:
            profile = svc.create(
                body.name,
                description=body.description,
                provider_id=body.provider_id,
                tools=body.tools,
                permission=body.permission or "workspace",
                prompt=body.prompt,
            )
        except AgentNameError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 409)
        return JSONResponse({"ok": True, "agent": _agent_payload(profile)})

    @app.get("/api/agents/{name}")
    def api_agents_get(name: str) -> JSONResponse:
        if not ctx.has("agents"):
            return JSONResponse({"ok": False, "error": "未加载 agents 插件"}, 503)
        try:
            profile = ctx.require("agents").get(name)
        except (KeyError, AgentNameError):
            return JSONResponse({"ok": False, "error": "agent 不存在"}, 404)
        return JSONResponse({"ok": True, "agent": _agent_payload(profile)})

    @app.put("/api/agents/{name}")
    def api_agents_update(name: str, body: AgentUpdateIn) -> JSONResponse:
        if not ctx.has("agents"):
            return JSONResponse({"ok": False, "error": "未加载 agents 插件"}, 503)
        svc = ctx.require("agents")
        try:
            profile = svc.update(
                name,
                description=body.description,
                provider_id=body.provider_id,
                tools=body.tools,
                permission=body.permission,
                prompt=body.prompt,
            )
        except (KeyError, AgentNameError):
            return JSONResponse({"ok": False, "error": "agent 不存在"}, 404)
        return JSONResponse({"ok": True, "agent": _agent_payload(profile)})

    @app.delete("/api/agents/{name}")
    def api_agents_delete(
        name: str,
        keep_files: bool = Query(True),
    ) -> JSONResponse:
        if not ctx.has("agents"):
            return JSONResponse({"ok": False, "error": "未加载 agents 插件"}, 503)
        try:
            ctx.require("agents").delete(name, keep_files=keep_files)
        except AgentNameError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        except KeyError:
            return JSONResponse({"ok": False, "error": "agent 不存在"}, 404)
        return JSONResponse({"ok": True, "name": name, "keep_files": keep_files})

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

    @app.get("/api/prompts")
    def api_prompts() -> dict[str, Any]:
        _sync_prompt_variant()
        if ctx.has("prompt_registry"):
            registry = ctx.require("prompt_registry")
            return {
                "variant": registry.active_variant(),
                "prompts": registry.snapshot(),
            }
        return {"variant": "cloud", "prompts": builtin_entries()}

    def _knowledge_or_error() -> tuple[Any, JSONResponse | None]:
        if ctx.has("knowledge"):
            return ctx.require("knowledge"), None
        return None, JSONResponse(
            {"ok": False, "error": "未加载 knowledge-base 插件"},
            503,
        )

    @app.get("/api/knowledge/status")
    def api_kb_status() -> dict[str, Any]:
        kb, err = _knowledge_or_error()
        if err is not None:
            return {
                "available": False,
                "reason": "未加载 knowledge-base 插件",
                "embedder": "",
                "doc_count": 0,
            }
        return kb.status()

    @app.get("/api/knowledge/documents")
    def api_kb_list() -> dict[str, Any]:
        kb, err = _knowledge_or_error()
        if err is not None:
            return {
                "available": False,
                "reason": "未加载 knowledge-base 插件",
                "documents": [],
            }
        docs = [
            {
                "id": item.id,
                "name": item.name,
                "source": item.source,
                "chunks": item.chunks,
                "created_at": item.created_at,
            }
            for item in kb.list_documents()
        ]
        return {
            "available": kb.available,
            "reason": kb.reason,
            "documents": docs,
        }

    @app.post("/api/knowledge/documents")
    def api_kb_upload(body: KnowledgeIngestIn) -> JSONResponse:
        kb, err = _knowledge_or_error()
        if err is not None:
            return err
        if not kb.available:
            return JSONResponse({"ok": False, "error": kb.reason or "知识库未启用"}, 503)
        name = body.name or "note.txt"
        try:
            if body.content_b64:
                data = base64.b64decode(body.content_b64)
                info = kb.ingest_bytes(name, data, source=name)
            else:
                info = kb.ingest(name, body.text or body.content, source=name)
        except Exception as exc:  # noqa: BLE001 — API 返回错误文本
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        return JSONResponse({"ok": True, **info})

    @app.delete("/api/knowledge/documents/{doc_id}")
    def api_kb_delete(doc_id: str) -> JSONResponse:
        kb, err = _knowledge_or_error()
        if err is not None:
            return err
        ok = kb.delete(doc_id)
        return JSONResponse({"ok": ok, "id": doc_id}, 200 if ok else 404)

    @app.post("/api/knowledge/search")
    def api_kb_search(body: KnowledgeSearchIn) -> JSONResponse:
        kb, err = _knowledge_or_error()
        if err is not None:
            return err
        query = str(body.query or "")
        top_k = body.top_k
        hits = kb.search(query, top_k=top_k)
        return JSONResponse(
            {
                "query": query,
                "text": kb.search_text(query, top_k=top_k),
                "hits": [
                    {
                        "text": item.text,
                        "source": item.source,
                        "score": item.score,
                        "doc_id": item.doc_id,
                    }
                    for item in hits
                ],
            }
        )

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

    @app.post("/api/config/reset")
    def api_config_reset() -> JSONResponse:
        """清空 settings.json，恢复为 pocket.yml / 内置默认并即时生效。"""
        with config_lock:
            if settings_file.is_file():
                try:
                    settings_file.unlink()
                except OSError as exc:
                    return JSONResponse(
                        {"ok": False, "error": f"无法删除 settings: {exc}"},
                        500,
                    )
            merged = _resolved_config()
            _apply_live_config(merged)
        return JSONResponse({"ok": True, "config": redact_secrets(merged.to_dict())})

    # ── Providers API（Phase 20 多模型）────────────────────

    def _load_providers_cfg() -> Any:
        from pocket_agent.providers import load_providers_config

        return load_providers_config(load_settings(settings_file), _resolved_config())

    def _save_providers_cfg(cfg: Any) -> None:
        from pocket_agent.providers import providers_to_dict

        with config_lock:
            current = load_settings(settings_file)
            data = providers_to_dict(cfg)
            current["providers"] = data["providers"]
            current["default_provider_id"] = data["default_provider_id"]
            current["routing"] = data["routing"]
            save_settings(current, settings_file)

    @app.get("/api/providers/presets")
    def api_providers_presets() -> dict[str, Any]:
        from pocket_agent.providers import preset_catalog

        return {"presets": preset_catalog()}

    @app.get("/api/providers")
    def api_providers_list() -> dict[str, Any]:
        from pocket_agent.providers import redact_providers

        cfg = _load_providers_cfg()
        return redact_providers(cfg)

    @app.post("/api/providers")
    def api_providers_create(body: ProviderIn) -> JSONResponse:
        import uuid as _uuid

        from pocket_agent.providers import ProviderConfig, apply_preset

        cfg = _load_providers_cfg()
        pid = _uuid.uuid4().hex[:8]
        preset = apply_preset(body.type)
        thinking = body.thinking if body.thinking in {"off", "strip", "keep"} else preset["thinking"]
        trust_env = (
            body.trust_env if body.trust_env is not None else bool(preset["trust_env"])
        )
        pc = ProviderConfig(
            id=pid,
            name=body.name,
            type=body.type,  # type: ignore[arg-type]
            base_url=body.base_url,
            api_key=body.api_key,
            model=body.model,
            sampling=body.sampling,
            thinking=thinking,
            trust_env=trust_env,
        )
        cfg.providers.append(pc)
        if not cfg.default_provider_id:
            cfg.default_provider_id = pid
        if not cfg.routing.choose_fill:
            cfg.routing.choose_fill = pid
        if not cfg.routing.generate:
            cfg.routing.generate = pid
        _save_providers_cfg(cfg)
        _sync_prompt_variant()
        from pocket_agent.providers import redact_providers

        return JSONResponse({"ok": True, "providers": redact_providers(cfg)["providers"]})

    @app.put("/api/providers/{pid}")
    def api_providers_update(pid: str, body: ProviderUpdateIn) -> JSONResponse:
        cfg = _load_providers_cfg()
        pc = next((p for p in cfg.providers if p.id == pid), None)
        if pc is None:
            return JSONResponse({"ok": False, "error": "provider not found"}, 404)
        if body.name is not None:
            pc.name = body.name
        if body.type is not None:
            pc.type = body.type  # type: ignore[assignment]
        if body.base_url is not None:
            pc.base_url = body.base_url
        if body.api_key is not None and "***" not in body.api_key:
            pc.api_key = body.api_key
        if body.model is not None:
            pc.model = body.model
        if body.sampling is not None:
            pc.sampling = body.sampling
        if body.thinking in {"off", "strip", "keep"}:
            pc.thinking = body.thinking  # type: ignore[assignment]
        if body.trust_env is not None:
            pc.trust_env = body.trust_env
        _save_providers_cfg(cfg)
        _sync_prompt_variant()
        from pocket_agent.providers import redact_providers

        return JSONResponse({"ok": True, "providers": redact_providers(cfg)["providers"]})

    @app.delete("/api/providers/{pid}")
    def api_providers_delete(pid: str) -> JSONResponse:
        cfg = _load_providers_cfg()
        if not any(p.id == pid for p in cfg.providers):
            return JSONResponse({"ok": False, "error": "provider not found"}, 404)
        cfg.providers = [p for p in cfg.providers if p.id != pid]
        if cfg.default_provider_id == pid:
            cfg.default_provider_id = cfg.providers[0].id if cfg.providers else ""
        if cfg.routing.choose_fill == pid:
            cfg.routing.choose_fill = cfg.default_provider_id
        if cfg.routing.generate == pid:
            cfg.routing.generate = cfg.default_provider_id
        _save_providers_cfg(cfg)
        _sync_prompt_variant()
        from pocket_agent.providers import redact_providers

        return JSONResponse({"ok": True, "providers": redact_providers(cfg)["providers"]})

    @app.post("/api/providers/{pid}/default")
    def api_providers_set_default(pid: str) -> JSONResponse:
        cfg = _load_providers_cfg()
        if not any(p.id == pid for p in cfg.providers):
            return JSONResponse({"ok": False, "error": "provider not found"}, 404)
        cfg.default_provider_id = pid
        _save_providers_cfg(cfg)
        _sync_prompt_variant()
        return JSONResponse({"ok": True, "default_provider_id": pid})

    @app.put("/api/routing")
    def api_routing_update(body: RoutingIn) -> JSONResponse:
        cfg = _load_providers_cfg()
        ids = {p.id for p in cfg.providers}
        if body.choose_fill not in ids or body.generate not in ids:
            return JSONResponse({"ok": False, "error": "provider id not found"}, 404)
        cfg.routing.choose_fill = body.choose_fill
        cfg.routing.generate = body.generate
        _save_providers_cfg(cfg)
        _sync_prompt_variant()
        return JSONResponse(
            {"ok": True, "routing": {"choose_fill": body.choose_fill, "generate": body.generate}}
        )

    # ── Workspace API（Phase 21 工作区）────────────────────

    @app.get("/api/workspaces")
    def api_workspaces_list() -> dict[str, Any]:
        ws = ctx.require("workspace") if ctx.has("workspace") else None
        if ws is None:
            return {"workspaces": [], "available": False}
        return {
            "workspaces": [
                {"id": w.id, "name": w.name, "root": w.root} for w in ws.list()
            ],
            "available": True,
        }

    @app.post("/api/workspaces")
    def api_workspaces_create(body: dict[str, Any]) -> JSONResponse:
        if not ctx.has("workspace"):
            return JSONResponse(
                {"ok": False, "error": "workspace plugin not mounted"}, 400
            )
        ws = ctx.require("workspace")
        name = str(body.get("name") or "").strip()
        root = str(body.get("root") or "").strip()
        if not name or not root:
            return JSONResponse(
                {"ok": False, "error": "name and root required"}, 400
            )
        created = ws.create(name, root)
        return JSONResponse(
            {"ok": True, "workspace": {"id": created.id, "name": created.name, "root": created.root}}
        )

    @app.put("/api/sessions/{session_id}/workspace")
    def api_session_bind_workspace(
        session_id: str, body: dict[str, Any]
    ) -> JSONResponse:
        if not ctx.has("workspace"):
            return JSONResponse(
                {"ok": False, "error": "workspace plugin not mounted"}, 400
            )
        ws = ctx.require("workspace")
        wid = str(body.get("workspace_id") or "").strip()
        if not wid:
            return JSONResponse({"ok": False, "error": "workspace_id required"}, 400)
        try:
            ws.bind_session(session_id, wid)
        except KeyError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 404)
        return JSONResponse({"ok": True, "session_id": session_id, "workspace_id": wid})

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

    def _abort_in_flight() -> None:
        if ctx.has("writer"):
            writer = ctx.require("writer")
            fn = getattr(writer, "request_stop", None)
            if callable(fn):
                fn()
        if ctx.has("sessions"):
            sessions = ctx.require("sessions")
            inner = getattr(sessions, "_writer", None)
            fn = getattr(inner, "request_stop", None) if inner is not None else None
            if callable(fn):
                fn()
            llm = getattr(sessions, "_llm", None)
            abort = getattr(llm, "abort_current", None) if llm is not None else None
            if callable(abort):
                abort()
        agent = ctx.require("agent") if ctx.has("agent") else None
        abort_agent = getattr(agent, "request_stop", None) if agent is not None else None
        if callable(abort_agent):
            abort_agent()
        if ctx.has("orchestrator"):
            orch = ctx.require("orchestrator")
            abort_orch = getattr(orch, "request_stop", None)
            if callable(abort_orch):
                abort_orch()

    @app.post("/api/chat/{run_id}/stop")
    def api_chat_stop(run_id: str) -> JSONResponse:
        runner = runs.get(run_id)
        if runner is not None:
            runner.request_stop()
        _abort_in_flight()
        if runner is None:
            return JSONResponse({"ok": False, "error": "run not found"}, 404)
        return JSONResponse({"ok": True, "run_id": run_id})

    def _sse_stream(
        run_fn: Callable[[], None],
        session_id: str,
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
                if not sse_accepts_session(session_id, payload):
                    return
                payload.setdefault("ts", time.time())
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
            join_worker_thread(thread, timeout=30)
            _release_lock()

    def _run_chat(
        *,
        message: str,
        session_id: str,
        run_id: str | None,
        overrides: dict[str, Any] | None,
        permission: str = "workspace",
        workspace_id: str | None = None,
        agent_name: str = "default",
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
        bound = _bind_routed_llm(live)
        _sync_prompt_variant(live)
        meta = _run_provider_meta(live)
        agent._run_meta = meta
        if ctx.has("sessions"):
            ctx.require("sessions")._run_meta = meta
        ctx.emit("run.start", {"run_id": rid, "session_id": session_id, **meta})
        runs[rid] = agent
        # overrides 中的 mode 是意图纠正，不是 AgentConfig 字段
        force_mode = None
        if overrides and "mode" in overrides:
            fm = str(overrides["mode"])
            if fm in ("qa", "agent", "writer"):
                force_mode = fm  # type: ignore[assignment]
        try:
            if ctx.has("sessions"):
                # run_id 只用于停止登记，会话路径始终走 SessionManager（意图路由）
                ctx.require("sessions").chat(
                    session_id,
                    message,
                    force_mode=force_mode,
                    permission=permission,
                    workspace_id=workspace_id,
                    agent=agent_name,
                )
            else:
                agent.run(message, run_id=run_id, session_id=session_id)
        finally:
            runs.pop(rid, None)
            for svc, llm in bound:
                svc._llm = llm
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
        run_id_s = str(body.run_id) if body.run_id else str(uuid.uuid4())
        overrides = dict(body.overrides or {})
        if ctx.has("agent"):
            runs[run_id_s] = ctx.require("agent")

        def _run() -> None:
            _run_chat(
                message=message,
                session_id=session_id,
                run_id=run_id_s,
                overrides=overrides,
                permission=str(body.permission or "workspace"),
                workspace_id=body.workspace_id,
                agent_name=str(body.agent or "default"),
            )

        return StreamingResponse(
            _sse_stream(_run, session_id),
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
        workspace_id: str | None = Query(None),
        permission: str = Query("workspace"),
        agent: str = Query("default"),
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
        rid = run_id or str(uuid.uuid4())
        if ctx.has("agent"):
            runs[rid] = ctx.require("agent")

        def _run() -> None:
            _run_chat(
                message=text,
                session_id=session_id,
                run_id=rid,
                overrides=parsed or None,
                permission=permission,
                workspace_id=workspace_id,
                agent_name=agent,
            )

        return StreamingResponse(
            _sse_stream(_run, session_id),
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
            _sse_stream(_run, "default"),
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

        sessions = ctx.require("sessions")
        if not getattr(sessions, "persist_enabled", False):
            raise PluginError("web 模式必须配置会话持久化目录")

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
        if _BUILD_INFO is not None:
            logger.info("ui-web build: %s (%s)", _BUILD_INFO.get("version"), _BUILD_INFO.get("built_at"))
        else:
            logger.warning(
                "前端产物缺失或过期，请执行: cd web && pnpm build"
            )
