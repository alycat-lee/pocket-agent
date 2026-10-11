from __future__ import annotations

import socket
import time
from typing import ClassVar

import pytest
from pocket_agent.plugins import builtin_plugins
from pocket_agent.plugins.ui_web import UiWebPlugin
from pocket_agent.runtime import Context, assemble

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _base_plugins_yml(port: int, *, profile: str) -> dict:
    return {
        "profile": profile,
        "plugins": [
            {"name": "llm-llamacpp", "config": {"base_url": "http://127.0.0.1:9/v1"}},
            {"name": "llm-deepseek", "disabled": True},
            {"name": "tools-core"},
            {
                "name": "agents",
                "config": {
                    "agents_root": ".agents_profile",
                    "legacy_sessions": ".no_legacy_sessions_profile",
                },
            },
            {"name": "checkpoint", "config": {"root": ".checkpoints_profile"}},
            {"name": "agent-loop", "config": {"max_steps": 3}},
            {"name": "session"},
            {"name": "writer"},
            {"name": "ui-cli"},
            {
                "name": "ui-web",
                "config": {"host": "127.0.0.1", "port": port},
            },
        ],
    }


def test_headless_profile_does_not_listen_port() -> None:
    port = _free_port()
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, _base_plugins_yml(port, profile="headless"))
    try:
        assert "ui-web" not in ctx.mount_order
        assert not ctx.has("http")
        assert not _port_open("127.0.0.1", port)
        assert ctx.has("agent")
        assert ctx.has("sessions")
    finally:
        ctx.unmount_all()


def test_web_profile_listens_and_unmount_releases_port() -> None:
    port = _free_port()
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, _base_plugins_yml(port, profile="web"))
    try:
        assert "ui-web" in ctx.mount_order
        assert ctx.has("http")
        http = ctx.require("http")
        assert http["port"] == port
        assert _port_open("127.0.0.1", port)
    finally:
        ctx.unmount_all()

    deadline = time.time() + 5
    while _port_open("127.0.0.1", port) and time.time() < deadline:
        time.sleep(0.05)
    assert not _port_open("127.0.0.1", port)


def test_ui_web_plugin_provides_http_service() -> None:
    port = _free_port()
    ctx = Context()

    class _Agent:
        approval_gate = None

        def apply_config(self, config: object) -> None:
            return None

        def request_stop(self) -> None:
            return None

    class _Sessions:
        persist_enabled = True

        def list_sessions(self) -> list:
            return []

    ctx.provide("agent", _Agent())
    ctx.provide("sessions", _Sessions())
    ctx.mount(UiWebPlugin(), {"host": "127.0.0.1", "port": port})
    try:
        assert ctx.has("http")
        assert _port_open("127.0.0.1", port)
    finally:
        ctx.unmount("ui-web")
    assert not _port_open("127.0.0.1", port)
