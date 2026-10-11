from __future__ import annotations

import ast
import operator
from datetime import datetime
from pathlib import Path
from typing import Any

from pocket_agent.tools import ToolRegistry, tool

_WEATHER: dict[str, str] = {
    "北京": "晴，25℃",
    "上海": "多云，18℃",
    "广州": "阵雨，27℃",
    "深圳": "阴，26℃",
    "成都": "多云，22℃",
}

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


def register_fixture_tools(registry: ToolRegistry) -> None:
    """测试用工具（含编造天气）。运行时禁止引用。"""

    @tool(registry, "查询指定城市天气")
    def get_weather(city: str) -> str:
        info = _WEATHER.get(city)
        if info is None:
            return f"{city}：暂无数据"
        return f"{city}：{info}"

    @tool(registry, "计算数学表达式")
    def calculator(expression: str) -> str:
        try:
            value = _safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 — 测试工具返回错误文本
            return f"计算失败：{exc}"
        return str(value)

    @tool(registry, "读取本地文件")
    def read_file(path: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            return f"文件不存在：{path}"
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"读取失败：{exc}"
        return text[:2000]

    @tool(registry, "获取当前时间")
    def get_current_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_eval(expression: str) -> float | int:
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
