"""提示词登记表：原子任务 → {id, template, variants, version}。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

PromptVariant = Literal["local", "cloud"]
Disposer = Callable[[], None]

_LOCAL_TYPES = frozenset({"llamacpp", "local"})


def variant_for_provider(provider_type: str) -> PromptVariant:
    """llamacpp / local → local；deepseek 等云端 preset → cloud。"""
    return "local" if (provider_type or "").strip().lower() in _LOCAL_TYPES else "cloud"


@dataclass
class PromptRecord:
    id: str
    variants: dict[str, str]
    version: str = "1.0.0"

    def resolve(self, variant: PromptVariant) -> str:
        if self.variants.get(variant):
            return self.variants[variant]
        return self.variants.get("cloud") or next(iter(self.variants.values()), "")

    def as_dict(self, variant: PromptVariant) -> dict[str, Any]:
        template = self.resolve(variant)
        return {
            "id": self.id,
            "template": template,
            "variants": dict(self.variants),
            "version": self.version,
        }


@dataclass
class PromptRegistry:
    """每个原子任务一条栈：后注册覆盖，disposer 弹出后恢复。"""

    _stacks: dict[str, list[PromptRecord]] = field(default_factory=dict)
    _variant: PromptVariant | None = None

    def register(
        self,
        prompt_id: str,
        *,
        variants: dict[str, str],
        version: str = "1.0.0",
        template: str | None = None,
    ) -> Disposer:
        cleaned = {str(k): str(v) for k, v in variants.items() if v is not None}
        if template and "cloud" not in cleaned:
            cleaned["cloud"] = template
        if not cleaned:
            raise ValueError(f"prompt {prompt_id!r} has no variants")
        record = PromptRecord(id=prompt_id, variants=cleaned, version=version)
        self._stacks.setdefault(prompt_id, []).append(record)
        disposed = False

        def dispose() -> None:
            nonlocal disposed
            if disposed:
                return
            disposed = True
            stack = self._stacks.get(prompt_id)
            if not stack:
                return
            if record in stack:
                stack.remove(record)
            if not stack:
                self._stacks.pop(prompt_id, None)

        return dispose

    def get(self, prompt_id: str) -> PromptRecord | None:
        stack = self._stacks.get(prompt_id)
        return stack[-1] if stack else None

    def resolve(self, prompt_id: str, *, fallback: str = "") -> str:
        record = self.get(prompt_id)
        if record is None:
            return fallback
        return record.resolve(self.active_variant())

    def list(self) -> list[PromptRecord]:
        return [stack[-1] for stack in self._stacks.values() if stack]

    def snapshot(self) -> list[dict[str, Any]]:
        variant = self.active_variant()
        return [record.as_dict(variant) for record in self.list()]

    def set_variant(self, variant: PromptVariant | None) -> None:
        self._variant = variant

    def select_for_provider(self, provider_type: str) -> PromptVariant:
        chosen = variant_for_provider(provider_type)
        self._variant = chosen
        return chosen

    def active_variant(self) -> PromptVariant:
        return self._variant or "cloud"


_active: PromptRegistry | None = None


def bind_registry(registry: PromptRegistry | None) -> None:
    global _active
    _active = registry


def current_registry() -> PromptRegistry | None:
    return _active


def resolve_prompt(prompt_id: str, *, fallback: str = "") -> str:
    if _active is None:
        return fallback
    return _active.resolve(prompt_id, fallback=fallback)
