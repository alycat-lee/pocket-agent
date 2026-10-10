from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from pocket_agent.config import AgentConfig

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class Tool:
    name: str
    short_desc: str  # ≤ 20 字，供 choose_tool 展示
    schema: dict
    fn: Callable[..., str]  # 执行函数，返回值必须为 str
    cacheable: bool = True  # False 表示写操作，跨轮账本持久拦截


class ToolRegistry:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        fn: Callable[..., str],
        short_desc: str | None = None,
        *,
        cacheable: bool = True,
    ) -> Tool:
        name = fn.__name__
        desc = (short_desc or "").strip() or name
        if len(desc) > 20:
            desc = desc[:20]
        tool = Tool(
            name=name,
            short_desc=desc,
            schema=_build_schema(fn),
            fn=fn,
            cacheable=cacheable,
        )
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool not found: {name}") from exc

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def recall(self, task: str, top_n: int) -> list[Tool]:
        """关键词召回：task 与 name+short_desc 做字符重叠打分，返回 top_n。
        工具总数 ≤ 阈值时直接返回全部。"""
        tools = self.all()
        if len(tools) <= self._config.tool_recall_threshold:
            return tools
        scored = [(_overlap_score(task, f"{t.name}{t.short_desc}"), t) for t in tools]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [t for _, t in scored[:top_n]]


def tool(
    registry: ToolRegistry,
    short_desc: str = "",
    *,
    cacheable: bool = True,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """装饰器：@tool(registry, "查询指定城市天气")"""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        registry.register(fn, short_desc=short_desc or None, cacheable=cacheable)
        return fn

    return decorator


def _build_schema(fn: Callable[..., str]) -> dict:
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = hints.get(name, str)
        properties[name] = {"type": _json_type(ann)}
        required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if args:
            return _json_type(args[0])
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    return "string"


def _overlap_score(task: str, text: str) -> float:
    if not task or not text:
        return 0.0
    task_chars = set(task.lower())
    text_chars = set(text.lower())
    if not task_chars:
        return 0.0
    overlap = len(task_chars & text_chars)
    # 子串加分
    bonus = 2.0 if text.lower() in task.lower() or task.lower() in text.lower() else 0.0
    return overlap / len(task_chars) + bonus
