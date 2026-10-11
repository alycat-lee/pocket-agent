from __future__ import annotations

from typing import ClassVar

import pytest
from pocket_agent.plugins import builtin_plugins
from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.llm_llamacpp import LlmLlamaCppPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.runtime import Context, PluginInUse, ServiceNotReady, assemble
from pocket_agent.tools import ToolRegistry


class _FakeLlm:
    def generate(self, prompt: str) -> str:
        return "ok"

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return -1

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return {}


class _FakeLlmPlugin:
    name = "llm-fake"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm"]

    def apply(self, ctx: Context, config: dict) -> None:
        ctx.provide("llm", _FakeLlm())


class _ToolsXPlugin:
    """第三方工具插件：向 tools 注册并挂 disposer。"""

    name = "tools-x"
    requires: ClassVar[list[str]] = ["tools"]
    provides: ClassVar[list[str]] = []

    def apply(self, ctx: Context, config: dict) -> None:
        registry: ToolRegistry = ctx.require("tools")

        def weather_extra(city: str) -> str:
            return f"extra:{city}"

        disposer = registry.register_with_disposer(
            weather_extra,
            "扩展天气查询工具",
        )
        ctx.on_dispose(disposer)


def test_unmount_tools_x_removes_tool_from_recall() -> None:
    ctx = Context()
    ctx.mount(ToolsCorePlugin())
    ctx.mount(_ToolsXPlugin())

    registry: ToolRegistry = ctx.require("tools")
    names = {t.name for t in registry.all()}
    assert "weather_extra" in names
    recalled = registry.recall("扩展天气", top_n=10)
    assert any(t.name == "weather_extra" for t in recalled)

    ctx.unmount("tools-x")
    names_after = {t.name for t in registry.all()}
    assert "weather_extra" not in names_after
    recalled_after = registry.recall("扩展天气", top_n=10)
    assert all(t.name != "weather_extra" for t in recalled_after)


def test_unmount_llm_cascades_agent_loop() -> None:
    ctx = Context()
    ctx.mount(_FakeLlmPlugin())
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin())

    assert "agent-loop" in ctx.mount_order
    assert ctx.has("agent")

    ctx.unmount("llm-fake")  # cascade 默认 True
    assert "agent-loop" not in ctx.mount_order
    assert not ctx.has("agent")
    with pytest.raises(ServiceNotReady):
        ctx.require("llm")


def test_unmount_llm_without_cascade_raises_plugin_in_use() -> None:
    ctx = Context()
    ctx.mount(_FakeLlmPlugin())
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin())

    with pytest.raises(PluginInUse):
        ctx.unmount("llm-fake", cascade=False)
    assert ctx.has("agent")


def test_headless_builtin_assemble_without_ui_web() -> None:
    """headless pocket.yml 装配成功且不含 ui-web。"""
    yml = {
        "profile": "headless",
        "plugins": [
            {
                "name": "llm-llamacpp",
                "config": {"base_url": "http://127.0.0.1:9/v1"},
            },
            {"name": "llm-deepseek", "disabled": True},
            {"name": "tools-core"},
            {"name": "checkpoint", "config": {"root": ".checkpoints_test_iso"}},
            {"name": "agent-loop", "config": {"max_steps": 3}},
            {"name": "session"},
            {"name": "writer"},
            {"name": "ui-cli"},
            {"name": "ui-web", "config": {"port": 3999}},
        ],
    }
    # 用假 LLM 替换真实 llamacpp，避免装配后连网
    plugins = builtin_plugins()
    plugins["llm-llamacpp"] = _FakeLlmAsLlama()  # type: ignore[assignment]
    ctx = assemble(plugins, yml)
    try:
        assert "ui-web" not in ctx.mount_order
        assert ctx.has("agent")
        assert ctx.has("tools")
        assert ctx.has("sessions")
        assert ctx.has("writer")
        assert ctx.require("sessions")._writer is not None
        assert ctx.has("checkpoint")
        result = ctx.require("agent").run("无需工具")
        assert result.answer == "ok"
    finally:
        ctx.unmount_all()


class _FakeLlmAsLlama:
    """占位：同名 llm-llamacpp，提供 llm 供装配测试。"""

    name = "llm-llamacpp"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm", "llm_local"]

    def apply(self, ctx: Context, config: dict) -> None:
        llm = _FakeLlm()
        ctx.provide("llm_local", llm)
        ctx.provide("llm", llm)


def test_routing_llm_picks_cloud_for_final_answer() -> None:
    from pocket_agent.plugins.llm_deepseek import RoutingLLM

    calls: list[str] = []

    class _Tagged:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def generate(self, prompt: str) -> str:
            calls.append(self.tag)
            return self.tag

        def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
            calls.append(self.tag)
            return minimum

        def generate_json(self, prompt: str, schema: dict) -> dict:
            calls.append(self.tag)
            return {}

    router = RoutingLLM(_Tagged("local"), _Tagged("cloud"))
    assert router.generate("请选择工具") == "local"
    assert router.generate("基于以上信息回答任务：天气") == "cloud"
    assert router.generate("你正在逐段撰写文章，第1段") == "cloud"
    assert calls == ["local", "cloud", "cloud"]


def test_deepseek_route_prepare_requires_llm_local() -> None:
    from pocket_agent.plugins.llm_deepseek import LlmDeepSeekPlugin

    p = LlmDeepSeekPlugin()
    p.prepare({"route": True, "api_key": "sk-test"})
    assert p.requires == ["llm_local"]

    ctx = Context()
    ctx.provide("llm_local", _FakeLlm())
    # provide outside plugin apply — ok for unit test
    p.apply(ctx, {"route": True, "api_key": "sk-test"})
    assert ctx.has("llm")
    from pocket_agent.plugins.llm_deepseek import RoutingLLM

    assert isinstance(ctx.require("llm"), RoutingLLM)


def test_llamacpp_local_only_does_not_provide_llm() -> None:
    # 不真正连网：替换 apply 内 client —— 仅测 provide 逻辑需 mock
    # 使用 prepare + 手动模拟 provide 路径
    plugin = LlmLlamaCppPlugin()
    plugin.prepare({"local_only": True})
    assert plugin.provides == ["llm_local"]

    class _StubClient(LlmLlamaCppPlugin):
        def apply(self, ctx: Context, config: dict) -> None:
            ctx.provide("llm_local", _FakeLlm())
            if bool(config.get("local_only")):
                self.provides = ["llm_local"]
                return
            ctx.provide("llm", _FakeLlm())

    ctx = Context()
    stub = _StubClient()
    stub.prepare({"local_only": True})
    ctx.mount(stub, {"local_only": True})
    assert ctx.has("llm_local")
    assert not ctx.has("llm")
