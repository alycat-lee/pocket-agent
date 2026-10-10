from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from pocket_agent.runtime import (
    Context,
    PluginError,
    PluginInUse,
    ServiceNotReady,
    assemble,
    load_pocket_config,
)


class _RecordingPlugin:
    def __init__(
        self,
        name: str,
        *,
        requires: list[str] | None = None,
        provides: list[str] | None = None,
        provide_keys: list[str] | None = None,
        fail: bool = False,
        on_apply: object | None = None,
    ) -> None:
        self.name = name
        self.requires = requires or []
        self.provides = provides if provides is not None else list(provide_keys or [])
        self._provide_keys = (
            provide_keys if provide_keys is not None else list(self.provides)
        )
        self._fail = fail
        self._on_apply = on_apply
        self.applied = False

    def apply(self, ctx: Context, config: dict) -> None:
        if self._on_apply is not None:
            self._on_apply(ctx, config)
        for key in self._provide_keys:
            ctx.provide(key, {"from": self.name, "key": key})
        if self._fail:
            raise RuntimeError(f"apply failed: {self.name}")
        self.applied = True


def test_mount_unmount_order() -> None:
    ctx = Context()
    order: list[str] = []

    class P:
        name = "p"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["s"]

        def apply(self, c: Context, config: dict) -> None:
            c.provide("s", 1)
            c.on_dispose(lambda: order.append("d1"))
            c.on_dispose(lambda: order.append("d2"))
            c.on("ping", lambda payload: order.append(f"evt:{payload['x']}"))

    ctx.mount(P())
    assert ctx.mount_order == ["p"]
    assert ctx.require("s") == 1
    ctx.emit("ping", {"x": 7})
    assert "evt:7" in order

    ctx.unmount("p")
    assert ctx.mount_order == []
    with pytest.raises(ServiceNotReady):
        ctx.require("s")
    dispose_calls = [x for x in order if x.startswith("d")]
    assert dispose_calls == ["d2", "d1"]


def test_deferred_activation_when_requires_not_ready() -> None:
    ctx = Context()
    consumer = _RecordingPlugin(
        "consumer",
        requires=["llm"],
        provides=[],
        provide_keys=[],
    )
    provider = _RecordingPlugin("provider", provides=["llm"], provide_keys=["llm"])

    ctx.mount(consumer)
    assert consumer.applied is False
    assert ctx.pending_names == ["consumer"]

    ctx.mount(provider)
    assert provider.applied is True
    assert consumer.applied is True
    assert ctx.pending_names == []
    assert ctx.mount_order == ["provider", "consumer"]
    assert ctx.require("llm")["from"] == "provider"


def test_disposers_run_in_reverse_on_unmount() -> None:
    ctx = Context()
    log: list[str] = []

    class P:
        name = "stack"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["a"]

        def apply(self, c: Context, config: dict) -> None:
            c.provide("a", "A")
            c.on_dispose(lambda: log.append("first"))
            c.on_dispose(lambda: log.append("second"))
            c.on_dispose(lambda: log.append("third"))

    ctx.mount(P())
    ctx.unmount("stack")
    assert log == ["third", "second", "first"]


def test_plugin_in_use_rejects_unmount() -> None:
    ctx = Context()
    ctx.mount(_RecordingPlugin("llm-p", provides=["llm"], provide_keys=["llm"]))
    ctx.mount(
        _RecordingPlugin(
            "agent",
            requires=["llm"],
            provides=["agent"],
            provide_keys=["agent"],
        )
    )
    with pytest.raises(PluginInUse):
        ctx.unmount("llm-p")
    assert ctx.has("llm")
    assert "llm-p" in ctx.mount_order

    ctx.unmount("agent")
    ctx.unmount("llm-p")
    assert ctx.mount_order == []


def test_apply_failure_full_rollback_in_assemble(tmp_path: Path) -> None:
    ok = _RecordingPlugin("ok", provides=["base"], provide_keys=["base"])
    bad = _RecordingPlugin(
        "bad",
        requires=["base"],
        provides=["x"],
        provide_keys=["x"],
        fail=True,
    )
    yml = tmp_path / "pocket.yml"
    yml.write_text(
        yaml.dump(
            {
                "profile": "headless",
                "plugins": [{"name": "ok"}, {"name": "bad"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="apply failed: bad"):
        assemble({"ok": ok, "bad": bad}, yml)

    assert ok.applied is True
    assert bad.applied is False


def test_assemble_apply_failure_clears_services() -> None:
    """装配失败后，不应残留已提供的服务。"""
    holder: dict[str, Context] = {}

    class Spy(Context):
        def unmount_all(self) -> None:
            holder["ctx"] = self
            super().unmount_all()

    ok = _RecordingPlugin("ok", provides=["base"], provide_keys=["base"])
    bad = _RecordingPlugin(
        "bad",
        requires=["base"],
        provides=["z"],
        provide_keys=["z"],
        fail=True,
    )
    yml = {
        "profile": "headless",
        "plugins": [{"name": "ok"}, {"name": "bad"}],
    }
    with pytest.raises(RuntimeError):
        assemble({"ok": ok, "bad": bad}, yml, ctx=Spy())
    ctx = holder["ctx"]
    assert ctx.mount_order == []
    with pytest.raises(ServiceNotReady):
        ctx.require("base")


def test_duplicate_provide_raises() -> None:
    ctx = Context()

    class P:
        name = "dup"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["s"]

        def apply(self, c: Context, config: dict) -> None:
            c.provide("s", 1)
            c.provide("s", 2)

    with pytest.raises(PluginError, match="already registered"):
        ctx.mount(P())
    assert ctx.mount_order == []


def test_require_not_ready() -> None:
    ctx = Context()
    with pytest.raises(ServiceNotReady):
        ctx.require("missing")


def test_provides_not_fulfilled_rolls_back() -> None:
    ctx = Context()

    class P:
        name = "liar"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["promised"]

        def apply(self, c: Context, config: dict) -> None:
            c.on_dispose(lambda: None)

    with pytest.raises(PluginError, match="did not provide"):
        ctx.mount(P())
    assert ctx.mount_order == []


def test_assemble_headless_skips_ui_web(tmp_path: Path) -> None:
    core = _RecordingPlugin("tools-core", provides=["tools"], provide_keys=["tools"])
    ui = _RecordingPlugin(
        "ui-web", requires=["tools"], provides=["http"], provide_keys=["http"]
    )
    yml = tmp_path / "pocket.yml"
    yml.write_text(
        yaml.dump(
            {
                "profile": "headless",
                "plugins": [{"name": "tools-core"}, {"name": "ui-web"}],
            }
        ),
        encoding="utf-8",
    )
    ctx = assemble({"tools-core": core, "ui-web": ui}, yml)
    assert core.applied is True
    assert ui.applied is False
    assert "ui-web" not in ctx.mount_order
    assert ctx.pending_names == []


def test_assemble_disabled_plugin_skipped() -> None:
    a = _RecordingPlugin("a", provides=["x"], provide_keys=["x"])
    b = _RecordingPlugin("b", provides=["y"], provide_keys=["y"])
    yml = {
        "profile": "headless",
        "plugins": [
            {"name": "a"},
            {"name": "b", "disabled": True},
        ],
    }
    ctx = assemble({"a": a, "b": b}, yml)
    assert a.applied is True
    assert b.applied is False
    assert ctx.mount_order == ["a"]


def test_assemble_passes_config() -> None:
    seen: dict[str, object] = {}

    class P:
        name = "cfg"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = ["c"]

        def apply(self, ctx: Context, config: dict) -> None:
            seen.update(config)
            ctx.provide("c", config)

    ctx = assemble(
        {"cfg": P()},
        {"profile": "headless", "plugins": [{"name": "cfg", "config": {"port": 9}}]},
    )
    assert seen == {"port": 9}
    assert ctx.require("c") == {"port": 9}


def test_load_pocket_config_from_file(tmp_path: Path) -> None:
    path = tmp_path / "pocket.yml"
    path.write_text("profile: web\nplugins:\n  - name: x\n", encoding="utf-8")
    data = load_pocket_config(path)
    assert data["profile"] == "web"
    assert data["plugins"][0]["name"] == "x"


def test_event_unsubscribe_on_unmount() -> None:
    ctx = Context()
    hits: list[int] = []

    class P:
        name = "ev"
        requires: ClassVar[list[str]] = []
        provides: ClassVar[list[str]] = []

        def apply(self, c: Context, config: dict) -> None:
            c.on("tick", lambda payload: hits.append(payload["n"]))

    ctx.mount(P())
    ctx.emit("tick", {"n": 1})
    assert hits == [1]
    ctx.unmount("ev")
    ctx.emit("tick", {"n": 2})
    assert hits == [1]
