from __future__ import annotations

import ast
import operator
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context
from pocket_agent.tools import ToolRegistry, tool

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def register_core_tools(registry: ToolRegistry) -> None:
    """注册真实执行的核心工具（不含编造数据）。"""

    @tool(registry, "计算数学表达式", domain="math")
    def calculator(expression: str) -> str:
        try:
            value = _safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 — 工具返回错误文本
            return f"计算失败：{exc}"
        return str(value)

    @tool(registry, "读取本地文件", domain="file")
    def read_file(path: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            return f"文件不存在：{path}"
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"读取失败：{exc}"
        return text[:2000]

    @tool(registry, "获取当前时间", domain="time")
    def get_current_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_eval(expression: str) -> float | int:
    """用 ast 安全求值四则运算，禁止 eval。"""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression: {type(node).__name__}")


class ToolsCorePlugin:
    """真实核心工具 → tools。"""

    name = "tools-core"
    requires: ClassVar[list[str]] = []
    provides: ClassVar[list[str]] = ["tools"]

    def apply(self, ctx: Context, config: dict) -> None:
        registry = ToolRegistry(agent_config_from(config))
        register_core_tools(registry)
        ctx.provide("tools", registry)
