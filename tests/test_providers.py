from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from pocket_agent.config import AgentConfig
from pocket_agent.plugins.ui_web import ApprovalBroker, build_app
from pocket_agent.providers import (
    PROVIDER_PRESETS,
    load_providers_config,
    migrate_providers,
)
from pocket_agent.runtime import Context

pytest.importorskip("fastapi")


def test_migrate_creates_providers_from_flat_fields() -> None:
    base = AgentConfig(base_url="http://localhost:8000/v1", model="test.gguf")
    settings: dict = {}
    migrated = migrate_providers(settings, base)
    assert "providers" in migrated
    assert len(migrated["providers"]) == 1
    assert migrated["providers"][0]["id"] == "local"
    assert migrated["providers"][0]["base_url"] == "http://localhost:8000/v1"
    assert migrated["default_provider_id"] == "local"
    assert migrated["routing"]["choose_fill"] == "local"


def test_migrate_idempotent() -> None:
    base = AgentConfig()
    settings = {"providers": [{"id": "custom", "name": "Keep Me", "api_key": "sk-keep"}]}
    migrated = migrate_providers(settings, base)
    item = migrated["providers"][0]
    assert item["id"] == "custom"
    assert item["name"] == "Keep Me"
    assert item["api_key"] == "sk-keep"
    assert item["thinking"] == PROVIDER_PRESETS["llamacpp"]["thinking"]
    assert item["trust_env"] is False


def test_migrate_upgrades_old_deepseek_without_dropping_data() -> None:
    base = AgentConfig(model="fallback-model")
    settings = {
        "providers": [
            {
                "id": "cloud",
                "name": "DeepSeek",
                "type": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-keep-me",
                "model": "deepseek-chat",
            }
        ],
        "default_provider_id": "cloud",
        "routing": {"choose_fill": "cloud", "generate": "cloud"},
    }
    migrated = migrate_providers(settings, base)
    item = migrated["providers"][0]
    assert item["api_key"] == "sk-keep-me"
    assert item["model"] == "deepseek-chat"
    assert item["base_url"] == "https://api.deepseek.com/v1"
    assert item["thinking"] == "strip"
    assert item["trust_env"] is True
    assert migrated["default_provider_id"] == "cloud"


def test_load_providers_config() -> None:
    base = AgentConfig(base_url="http://localhost:8000/v1")
    settings = {
        "providers": [
            {"id": "p1", "name": "Local", "type": "llamacpp", "base_url": "http://x/v1"},
            {"id": "p2", "name": "Cloud", "type": "deepseek", "base_url": "https://api.deepseek.com/v1"},
        ],
        "default_provider_id": "p1",
        "routing": {"choose_fill": "p1", "generate": "p2"},
    }
    cfg = load_providers_config(settings, base)
    assert len(cfg.providers) == 2
    assert cfg.providers[0].id == "p1"
    assert cfg.routing.choose_fill == "p1"
    assert cfg.routing.generate == "p2"


async def _request(app, method, path, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        return await c.request(method, path, **kwargs)


def test_providers_crud_api(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    ctx = Context()
    from pocket_agent.plugins.agent_loop import AgentLoopPlugin
    from pocket_agent.plugins.session_plugin import SessionPlugin
    from pocket_agent.plugins.tools_core import ToolsCorePlugin

    class _L:
        def generate(self, p): return "x"
        def generate_int(self, p, mn, mx): return -1
        def generate_json(self, p, s): return {}

    ctx.provide("llm", _L())
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin())
    ctx.mount(SessionPlugin())
    app = build_app(
        ctx,
        broker=ApprovalBroker(1),
        settings_path=settings,
        pocket_data={"plugins": []},
    )

    # Create (auto-migration adds a 'local' provider first)
    res = asyncio.run(
        _request(app, "POST", "/api/providers", json={
            "name": "Local", "type": "llamacpp", "base_url": "http://localhost:8000/v1"
        })
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    providers = body["providers"]
    # 1 migrated + 1 new
    assert len(providers) == 2
    pid = providers[-1]["id"]  # newly created

    # List
    res = asyncio.run(_request(app, "GET", "/api/providers"))
    assert res.status_code == 200
    assert len(res.json()["providers"]) == 2

    # Update
    res = asyncio.run(
        _request(app, "PUT", f"/api/providers/{pid}", json={"name": "Local Updated"})
    )
    assert res.status_code == 200
    updated = [p for p in res.json()["providers"] if p["id"] == pid]
    assert updated and updated[0]["name"] == "Local Updated"

    # Routing
    res = asyncio.run(
        _request(app, "PUT", "/api/routing", json={"choose_fill": pid, "generate": pid})
    )
    assert res.status_code == 200
    assert res.json()["routing"]["choose_fill"] == pid

    # Delete (leaves the migrated 'local' provider)
    res = asyncio.run(_request(app, "DELETE", f"/api/providers/{pid}"))
    assert res.status_code == 200
    remaining = res.json()["providers"]
    assert len(remaining) == 1
    assert remaining[0]["id"] != pid

    ctx.unmount_all()


def test_providers_presets_api(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    ctx = Context()
    app = build_app(
        ctx,
        broker=ApprovalBroker(1),
        settings_path=settings,
        pocket_data={"plugins": []},
    )
    res = asyncio.run(_request(app, "GET", "/api/providers/presets"))
    assert res.status_code == 200
    ids = {p["id"] for p in res.json()["presets"]}
    assert ids == {"llamacpp", "deepseek", "openai_compatible"}
    ctx.unmount_all()
