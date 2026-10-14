from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from pocket_agent.config import AgentConfig

Disposer = Callable[[], None]

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
    domain: str = "general"


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
        domain: str = "general",
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
            domain=domain or "general",
        )
        self._tools[name] = tool
        return tool

    def register_with_disposer(
        self,
        tool: Tool | Callable[..., str],
        short_desc: str | None = None,
        *,
        cacheable: bool = True,
        domain: str = "general",
    ) -> Disposer:
        """注册工具并返回注销函数（插件卸载时回收）。"""
        if isinstance(tool, Tool):
            self._tools[tool.name] = tool
            name = tool.name
        else:
            registered = self.register(
                tool, short_desc, cacheable=cacheable, domain=domain
            )
            name = registered.name

        disposed = False

        def dispose() -> None:
            nonlocal disposed
            if disposed:
                return
            disposed = True
            self._tools.pop(name, None)

        return dispose

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

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


_DIRECT_DOMAINS = frozenset({"weather", "time", "file", "shell", "math"})
_KNOWLEDGE_DOMAINS = frozenset({"knowledge", "web"})
_DOMAIN_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weather", ("天气", "气温", "温度", "下雨", "多少度", "晴", "阴")),
    ("time", ("几点", "几点了", "当前时间", "星期几", "现在几点")),
    ("math", ("计算", "温差", "平方", "相加", "相减", "乘以", "除以", "等于几")),
    ("file", ("读取文件", "写入文件", "编辑文件", "列出目录", "工作区文件")),
    ("shell", ("运行命令", "执行命令", "终端命令")),
    ("knowledge", ("百科", "维基", "维基百科", "知识库", "网页搜索", "网上搜", "搜索网页")),
    ("web", ("新闻", "头条", "要闻", "联网", "今日新闻", "实时新闻")),
)


def tool(
    registry: ToolRegistry,
    short_desc: str = "",
    *,
    cacheable: bool = True,
    domain: str = "general",
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """装饰器：@tool(registry, "查询指定城市天气")"""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        registry.register(
            fn,
            short_desc=short_desc or None,
            cacheable=cacheable,
            domain=domain,
        )
        return fn

    return decorator


def detect_tool_domains(task: str) -> set[str]:
    body = task or ""
    return {name for name, hints in _DOMAIN_HINTS if any(h in body for h in hints)}


def tool_name_domain(name: str) -> str:
    """按工具名推断 domain，供 verifier 在无注册表时判断领域覆盖。"""
    lowered = (name or "").strip().lower()
    if "weather" in lowered:
        return "weather"
    if lowered in {"get_current_time"} or lowered.endswith("_time"):
        return "time"
    if lowered == "calculator" or "calc" in lowered:
        return "math"
    if any(
        token in lowered
        for token in ("file", "list_dir", "grep")
    ):
        return "file"
    if "command" in lowered:
        return "shell"
    if lowered in {"search_knowledge", "wiki_lookup"}:
        return "knowledge"
    if lowered == "web_search":
        return "web"
    return "general"


def filter_tools_by_domain(task: str, tools: list[Tool]) -> list[Tool]:
    """命中直接数据领域且未命中知识领域时，去掉 knowledge/web；不得返回空表。"""
    if not tools:
        return []
    domains = detect_tool_domains(task)
    if domains & _DIRECT_DOMAINS and not (domains & _KNOWLEDGE_DOMAINS):
        kept = [t for t in tools if t.domain not in _KNOWLEDGE_DOMAINS]
        if kept:
            return kept
    return list(tools)


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
