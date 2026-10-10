from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

Disposer = Callable[[], None]

# headless profile 默认跳过的 UI 插件名
_HEADLESS_SKIP = frozenset({"ui-web"})


class PluginError(Exception):
    """插件运行时通用错误。"""


class ServiceNotReady(PluginError):
    """require 的服务尚未注册。"""


class PluginInUse(PluginError):
    """仍有其他活跃插件依赖本插件提供的服务，拒绝卸载。"""


class Plugin(Protocol):
    name: str
    requires: list[str]
    provides: list[str]

    def apply(self, ctx: Context, config: dict) -> None: ...


@dataclass
class _PluginRecord:
    plugin: Plugin
    config: dict[str, Any]
    disposers: list[Disposer] = field(default_factory=list)
    active: bool = False


class Context:
    """插件共享上下文。服务注册表 + 类型化事件总线 + 插件树管理。"""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._service_owners: dict[str, str] = {}
        self._handlers: dict[str, list[Callable[..., Any]]] = {}
        self._plugins: dict[str, _PluginRecord] = {}
        self._mount_order: list[str] = []
        self._pending: list[tuple[Plugin, dict[str, Any]]] = []
        self._current: str | None = None

    # ── 服务 ──────────────────────────────────────────────

    def provide(self, key: str, service: Any) -> Disposer:
        """注册服务，返回撤销函数。同 key 重复注册抛 PluginError。"""
        if key in self._services:
            raise PluginError(f"service already registered: {key}")
        self._services[key] = service
        owner = self._current
        if owner is not None:
            self._service_owners[key] = owner

        disposed = False

        def dispose() -> None:
            nonlocal disposed
            if disposed:
                return
            disposed = True
            self._services.pop(key, None)
            self._service_owners.pop(key, None)

        self.on_dispose(dispose)
        # 不在 apply 中途激活挂起插件，避免嵌套 _current；等当前插件 apply 结束再调度
        if self._current is None:
            self._try_activate_pending()
        return dispose

    def require(self, key: str) -> Any:
        """获取依赖服务，未就绪抛 ServiceNotReady。"""
        if key not in self._services:
            raise ServiceNotReady(f"service not ready: {key}")
        return self._services[key]

    def has(self, key: str) -> bool:
        return key in self._services

    # ── 事件 ──────────────────────────────────────────────

    def on(self, event: str, handler: Callable[..., Any]) -> Disposer:
        """订阅事件，返回退订函数。"""
        bucket = self._handlers.setdefault(event, [])
        bucket.append(handler)
        disposed = False

        def dispose() -> None:
            nonlocal disposed
            if disposed:
                return
            disposed = True
            handlers = self._handlers.get(event)
            if handlers is None:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return
            if not handlers:
                self._handlers.pop(event, None)

        self.on_dispose(dispose)
        return dispose

    def emit(self, event: str, payload: dict) -> None:
        for handler in list(self._handlers.get(event, [])):
            handler(payload)

    # ── 回收栈 ────────────────────────────────────────────

    def on_dispose(self, disposer: Disposer) -> None:
        """把 disposer 登记到当前正在 apply 的插件回收栈。"""
        if self._current is None:
            return
        record = self._plugins.get(self._current)
        if record is None:
            return
        record.disposers.append(disposer)

    # ── 生命周期 ──────────────────────────────────────────

    def mount(self, plugin: Plugin, config: dict | None = None) -> None:
        """挂载插件：requires 就绪则 apply；否则挂起等待依赖。"""
        cfg = dict(config or {})
        name = plugin.name
        if name in self._plugins and self._plugins[name].active:
            raise PluginError(f"plugin already mounted: {name}")
        if any(p.name == name for p, _ in self._pending):
            raise PluginError(f"plugin already pending: {name}")

        if not self._deps_ready(plugin.requires):
            self._pending.append((plugin, cfg))
            return

        self._activate(plugin, cfg)

    def unmount(self, name: str, *, cascade: bool = True) -> None:
        """逆序执行 disposer。

        cascade=True（默认）：先停用依赖本插件服务的活跃插件（消失则停用）。
        cascade=False：若仍有依赖方则抛 PluginInUse。
        """
        pending_hit = [p for p, _ in self._pending if p.name == name]
        if pending_hit:
            self._pending = [(p, c) for p, c in self._pending if p.name != name]
            return

        record = self._plugins.get(name)
        if record is None or not record.active:
            raise PluginError(f"plugin not mounted: {name}")

        provided = {
            key for key, owner in self._service_owners.items() if owner == name
        }
        provided.update(record.plugin.provides)
        dependents = [
            other_name
            for other_name, other in self._plugins.items()
            if other_name != name
            and other.active
            and any(req in provided for req in other.plugin.requires)
        ]
        if dependents and not cascade:
            raise PluginInUse(
                f"plugin {name!r} still required by {dependents!r}"
            )
        # 按挂载逆序停用依赖方
        for dep in reversed(self._mount_order):
            if dep in dependents:
                self.unmount(dep, cascade=True)

        # 重新取 record（级联过程不应删掉自己）
        record = self._plugins.get(name)
        if record is None or not record.active:
            return
        self._teardown(name, record)

    @property
    def mount_order(self) -> list[str]:
        """当前活跃插件的挂载顺序（只读副本）。"""
        return list(self._mount_order)

    @property
    def pending_names(self) -> list[str]:
        return [p.name for p, _ in self._pending]

    def unmount_all(self) -> None:
        """逆序强制卸载全部活跃插件（装配失败回滚用，忽略 PluginInUse）。"""
        for name in reversed(list(self._mount_order)):
            record = self._plugins.get(name)
            if record is not None and record.active:
                self._teardown(name, record)
        self._pending.clear()

    # ── 内部 ──────────────────────────────────────────────

    def _deps_ready(self, requires: list[str]) -> bool:
        return all(key in self._services for key in requires)

    def _activate(self, plugin: Plugin, config: dict[str, Any]) -> None:
        name = plugin.name
        record = _PluginRecord(plugin=plugin, config=config)
        self._plugins[name] = record
        self._current = name
        try:
            plugin.apply(self, config)
            for key in plugin.provides:
                if key not in self._services:
                    raise PluginError(
                        f"plugin {name!r} declared provides {key!r} but did not provide it"
                    )
                if self._service_owners.get(key) != name:
                    self._service_owners[key] = name
        except BaseException:
            self._run_disposers(record)
            self._plugins.pop(name, None)
            self._current = None
            raise
        self._current = None
        record.active = True
        self._mount_order.append(name)
        self._try_activate_pending()

    def _try_activate_pending(self) -> None:
        # 先移出 pending 再 activate，避免重入时重复激活同一插件
        while True:
            ready: tuple[Plugin, dict[str, Any]] | None = None
            for plugin, config in self._pending:
                if self._deps_ready(plugin.requires):
                    ready = (plugin, config)
                    break
            if ready is None:
                return
            plugin, config = ready
            self._pending = [
                (p, c) for p, c in self._pending if p.name != plugin.name
            ]
            self._activate(plugin, config)

    def _teardown(self, name: str, record: _PluginRecord) -> None:
        self._run_disposers(record)
        record.active = False
        if name in self._mount_order:
            self._mount_order.remove(name)
        self._plugins.pop(name, None)

    @staticmethod
    def _run_disposers(record: _PluginRecord) -> None:
        while record.disposers:
            disposer = record.disposers.pop()
            disposer()


def load_pocket_config(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """读取 pocket.yml（或已解析的 dict）。"""
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PluginError("pocket.yml root must be a mapping")
    return data


def assemble(
    plugins: Mapping[str, Plugin],
    config: str | Path | Mapping[str, Any],
    *,
    ctx: Context | None = None,
) -> Context:
    """按 pocket.yml 装配：按序挂载；失败则全量逆序回滚后抛出。"""
    data = load_pocket_config(config)
    profile = str(data.get("profile", "headless"))
    entries = data.get("plugins", [])
    if not isinstance(entries, list):
        raise PluginError("pocket.yml plugins must be a list")

    context = ctx if ctx is not None else Context()
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                raise PluginError(f"invalid plugin entry: {entry!r}")
            if entry.get("disabled"):
                continue
            name = entry.get("name")
            if not name or not isinstance(name, str):
                raise PluginError(f"plugin entry missing name: {entry!r}")
            if profile == "headless" and name in _HEADLESS_SKIP:
                continue
            if name not in plugins:
                raise PluginError(f"unknown plugin: {name}")
            plugin_cfg = entry.get("config") or {}
            if not isinstance(plugin_cfg, dict):
                raise PluginError(f"plugin {name!r} config must be a mapping")
            plugin = plugins[name]
            prepare = getattr(plugin, "prepare", None)
            if callable(prepare):
                prepare(plugin_cfg)
            context.mount(plugin, plugin_cfg)
    except BaseException:
        context.unmount_all()
        raise
    return context
