from __future__ import annotations

from typing import ClassVar

import pytest
from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.prompts_plugin import PromptsPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.prompt_registry import (
    PromptRegistry,
    bind_registry,
    current_registry,
    variant_for_provider,
)
from pocket_agent.prompts import CHOOSE_TOOL, SYSTEM_BASE, builtin_entries
from pocket_agent.runtime import Context


@pytest.fixture(autouse=True)
def _unbind_registry() -> None:
    bind_registry(None)
    yield
    bind_registry(None)


def test_variant_for_provider_preset() -> None:
    assert variant_for_provider("llamacpp") == "local"
    assert variant_for_provider("local") == "local"
    assert variant_for_provider("deepseek") == "cloud"
    assert variant_for_provider("openai_compatible") == "cloud"


def test_register_override_and_dispose_restores() -> None:
    registry = PromptRegistry()
    first = registry.register(
        "choose_tool",
        variants={"cloud": "FIRST {task}", "local": "FIRST-L {task}"},
        version="1",
    )
    second = registry.register(
        "choose_tool",
        variants={"cloud": "SECOND {task}", "local": "SECOND-L {task}"},
        version="2",
    )
    registry.set_variant("cloud")
    assert registry.resolve("choose_tool") == "SECOND {task}"
    assert registry.get("choose_tool") is not None
    assert registry.get("choose_tool").version == "2"
    second()
    assert registry.resolve("choose_tool") == "FIRST {task}"
    first()
    assert registry.get("choose_tool") is None


def test_local_variants_are_short_and_cloud_keeps_doc04() -> None:
    for entry in builtin_entries():
        local = str(entry["variants"]["local"])
        cloud = str(entry["variants"]["cloud"])
        assert len(local) <= 150, f"{entry['id']} local={len(local)}"
        assert cloud == entry["template"]
    assert "下一步应该使用哪个工具" in str.__str__(CHOOSE_TOOL)
    assert "通过调用工具逐步完成任务" in str.__str__(SYSTEM_BASE)


def test_promptref_uses_fallback_without_registry() -> None:
    bind_registry(None)
    text = CHOOSE_TOOL.format(
        task="t",
        state="s",
        history_text="（无）",
        options="0. calculator: 计算",
    )
    assert text.startswith("任务：t")
    assert "下一步应该使用哪个工具" in text


def test_promptref_follows_active_variant() -> None:
    registry = PromptRegistry()
    registry.register(
        "choose_tool",
        variants={"cloud": "CLOUD {task}", "local": "LOCAL {task}"},
        version="t",
    )
    bind_registry(registry)
    registry.set_variant("cloud")
    assert CHOOSE_TOOL.format(task="X", state="", history_text="", options="") == "CLOUD X"
    registry.select_for_provider("llamacpp")
    assert CHOOSE_TOOL.format(task="X", state="", history_text="", options="") == "LOCAL X"
    registry.select_for_provider("deepseek")
    assert CHOOSE_TOOL.format(task="X", state="", history_text="", options="") == "CLOUD X"


class _FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "完成"

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        self.prompts.append(prompt)
        return -1

    def generate_json(self, prompt: str, schema: dict) -> dict:
        self.prompts.append(prompt)
        return {}


class _FakeLlmPlugin:
    name = "llm-fake"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["llm"]

    def __init__(self, llm: _FakeLlm) -> None:
        self._llm = llm

    def apply(self, ctx: Context, config: dict) -> None:
        del config
        ctx.provide("llm", self._llm)


class _CustomChoosePlugin:
    name = "prompts-custom-choose"
    requires: ClassVar[list[str]] = ["prompt_registry"]
    provides: ClassVar[list[str]] = []

    def apply(self, ctx: Context, config: dict) -> None:
        del config
        registry = ctx.require("prompt_registry")
        disposer = registry.register(
            "choose_tool",
            variants={
                "cloud": (
                    "CUSTOM_CHOOSE_MARK 任务：{task}\n进度：{state}\n"
                    "历史：{history_text}\n工具：{options}\n只输出一个数字。例如：0"
                ),
                "local": "CUSTOM_CHOOSE_MARK {task} {state} {history_text} {options} 只输出数字。例如：0",
            },
            version="custom-1",
        )
        ctx.on_dispose(disposer)


def _mount_agent(ctx: Context, llm: _FakeLlm) -> None:
    ctx.mount(PromptsPlugin())
    ctx.mount(_FakeLlmPlugin(llm))
    ctx.mount(ToolsCorePlugin())
    ctx.mount(AgentLoopPlugin(), {"max_steps": 2, "planner_enabled": False})


def test_custom_choose_plugin_changes_llm_prompt_then_unmount_restores() -> None:
    ctx = Context()
    llm = _FakeLlm()
    _mount_agent(ctx, llm)
    ctx.mount(_CustomChoosePlugin())
    assert current_registry() is not None
    assert current_registry().get("choose_tool").version == "custom-1"

    ctx.require("agent").run("现在几点了")
    assert any("CUSTOM_CHOOSE_MARK" in p for p in llm.prompts)

    ctx.unmount("prompts-custom-choose")
    assert current_registry().get("choose_tool").version == "1.0.0"

    llm.prompts.clear()
    ctx.require("agent").run("现在几点了")
    assert all("CUSTOM_CHOOSE_MARK" not in p for p in llm.prompts)
    assert any("下一步应该使用哪个工具" in p for p in llm.prompts)
    ctx.unmount_all()


def test_prompts_plugin_unmount_removes_service() -> None:
    ctx = Context()
    ctx.mount(PromptsPlugin())
    assert ctx.has("prompt_registry")
    assert current_registry() is not None
    ctx.unmount("prompts")
    assert not ctx.has("prompt_registry")
    assert current_registry() is None


def test_promptref_slice_stays_plain_str() -> None:
    assert SYSTEM_BASE[:4] == "你是一个"
    assert isinstance(SYSTEM_BASE[:4], str)
    assert not isinstance(SYSTEM_BASE[:4], type(SYSTEM_BASE))


def test_builtin_plugins_includes_prompts() -> None:
    from pocket_agent.plugins import builtin_plugins

    assert "prompts" in builtin_plugins()
