from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from pocket_agent.config import load_settings
from pocket_agent.events import EventStore
from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.ui_web import (
    ApprovalBroker,
    attach_event_persistence,
    build_app,
)
from pocket_agent.runtime import Context

pytest.importorskip("fastapi")


class _FakeLlm:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)

    def _next(self) -> Any:
        return self._script.pop(0) if self._script else -1

    def generate(self, prompt: str) -> str:
        return str(self._next())

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return int(self._next())

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


def _stack(ctx: Context, script: list[Any]) -> None:
    ctx.mount(_FakeLlmPlugin(script))
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 8})
    ctx.mount(SessionPlugin())


async def _request(
    app: Any,
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=json_body)


async def _stream_post(app: Any, path: str, payload: dict) -> str:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", path, json=payload) as response,
    ):
        chunks: list[str] = []
        async for text in response.aiter_text():
            chunks.append(text)
        return "".join(chunks)


def test_put_config_persists_and_applies(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    store = EventStore(tmp_path / "events")
    ctx = Context()
    _stack(ctx, [-1, "hi"])
    attach_event_persistence(ctx, store)
    app = build_app(
        ctx,
        broker=ApprovalBroker(2),
        event_store=store,
        settings_path=settings,
        pocket_data={"plugins": [{"name": "agent-loop", "config": {"temperature": 0.1}}]},
    )

    res = asyncio.run(_request(app, "GET", "/api/config"))
    assert res.status_code == 200
    assert res.json()["temperature"] == 0.1

    res = asyncio.run(
        _request(app, "PUT", "/api/config", json_body={"temperature": 0.55})
    )
    assert res.status_code == 200
    assert abs(res.json()["config"]["temperature"] - 0.55) < 1e-9
    saved = load_settings(settings)
    assert saved["temperature"] == 0.55
    assert abs(ctx.require("agent")._config.temperature - 0.55) < 1e-9

    # overrides 不写 settings
    body = "".join(
        asyncio.run(
            _stream_post(
                app,
                "/api/chat",
                {
                    "message": "无需工具",
                    "session_id": "ov",
                    "overrides": {"temperature": 0.99},
                },
            )
        )
    )
    assert "answer.done" in body
    assert load_settings(settings)["temperature"] == 0.55
    assert abs(ctx.require("agent")._config.temperature - 0.55) < 1e-9
    ctx.unmount_all()


def test_api_key_redacted_on_get(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"api_key": "sk-super-secret"}),
        encoding="utf-8",
    )
    ctx = Context()
    _stack(ctx, [-1, "x"])
    app = build_app(
        ctx,
        broker=ApprovalBroker(1),
        settings_path=settings,
    )
    res = asyncio.run(_request(app, "GET", "/api/config"))
    assert res.json()["api_key"] == "sk-***"
    assert "super-secret" not in res.text
    ctx.unmount_all()


def test_schema_endpoint(tmp_path: Path) -> None:
    ctx = Context()
    _stack(ctx, [-1, "x"])
    app = build_app(ctx, broker=ApprovalBroker(1), settings_path=tmp_path / "s.json")
    res = asyncio.run(_request(app, "GET", "/api/config/schema"))
    assert res.status_code == 200
    keys = {item["key"] for item in res.json()}
    assert "temperature" in keys
    # deprecated 项不再返回（Phase 22：schema 剔除废弃项）
    assert "api_key" not in keys
    assert "base_url" not in keys
    assert "model" not in keys
    assert "route_cloud" not in keys
    ctx.unmount_all()


def test_config_reset_clears_settings(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"max_steps": 3, "temperature": 0.9}),
        encoding="utf-8",
    )
    ctx = Context()
    _stack(ctx, [-1, "x"])
    app = build_app(
        ctx,
        broker=ApprovalBroker(1),
        settings_path=settings,
        pocket_data={
            "plugins": [{"name": "agent-loop", "config": {"max_steps": 10}}]
        },
    )
    res = asyncio.run(_request(app, "POST", "/api/config/reset"))
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["config"]["max_steps"] == 10
    assert not settings.is_file()
    assert ctx.require("agent")._config.max_steps == 10
    ctx.unmount_all()


def test_events_api_after_chat(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events")
    ctx = Context()
    _stack(
        ctx,
        [0, {"city": "北京"}, -1, "晴"],
    )
    attach_event_persistence(ctx, store)
    app = build_app(
        ctx,
        broker=ApprovalBroker(2),
        event_store=store,
        settings_path=tmp_path / "s.json",
    )
    asyncio.run(
        _stream_post(
            app,
            "/api/chat",
            {"message": "查北京", "session_id": "default"},
        )
    )
    res = asyncio.run(
        _request(app, "GET", "/api/sessions/default/events?limit=20")
    )
    assert res.status_code == 200
    data = res.json()
    assert data["events"]
    names = [e["event"] for e in data["events"]]
    assert "step.start" in names
    assert "answer.done" in names
    # 字段完整性
    step_ev = next(e for e in data["events"] if e["event"] == "step.start")
    payload = step_ev["data"]
    for key in ("step_index", "elapsed_ms", "prompt_tokens_est", "ttft_ms"):
        assert key in payload

    export = asyncio.run(_request(app, "GET", "/api/sessions/default/export"))
    assert export.status_code == 200
    assert "warning" in export.json()
    assert export.json()["events"]
    ctx.unmount_all()


def test_plugins_list(tmp_path: Path) -> None:
    ctx = Context()
    _stack(ctx, [-1, "x"])
    from pocket_agent.plugins.tools_core import ToolsCorePlugin

    app = build_app(
        ctx,
        broker=ApprovalBroker(1),
        settings_path=tmp_path / "s.json",
        plugin_catalog={"tools-core": ToolsCorePlugin(), "llm-fake": _FakeLlmPlugin([])},
    )
    res = asyncio.run(_request(app, "GET", "/api/plugins"))
    assert res.status_code == 200
    rows = {r["name"]: r for r in res.json()}
    assert rows["tools-core"]["enabled"] is True
    ctx.unmount_all()
