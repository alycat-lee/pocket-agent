from __future__ import annotations

from pathlib import Path

from pocket_agent.config import AgentConfig
from pocket_agent.plugins.tools_core import register_core_tools
from pocket_agent.tools import ToolRegistry, tool

from fixtures.fixture_tools import register_fixture_tools


def test_schema_generation_four_types() -> None:
    registry = ToolRegistry()

    @tool(registry, "类型示例")
    def typed(a: str, b: int, c: float, d: bool) -> str:
        return f"{a}{b}{c}{d}"

    schema = registry.get("typed").schema
    assert schema["type"] == "object"
    assert schema["properties"]["a"]["type"] == "string"
    assert schema["properties"]["b"]["type"] == "integer"
    assert schema["properties"]["c"]["type"] == "number"
    assert schema["properties"]["d"]["type"] == "boolean"
    assert set(schema["required"]) == {"a", "b", "c", "d"}


def test_decorator_registers_tool() -> None:
    registry = ToolRegistry()

    @tool(registry, "查询指定城市天气")
    def get_weather(city: str) -> str:
        return city

    tools = registry.all()
    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].short_desc == "查询指定城市天气"
    assert tools[0].fn("北京") == "北京"


def test_recall_ranking_and_threshold() -> None:
    config = AgentConfig(tool_recall_threshold=2, tool_recall_top_n=1)
    registry = ToolRegistry(config)

    @tool(registry, "查询天气")
    def get_weather(city: str) -> str:
        return city

    @tool(registry, "做计算")
    def calculator(expression: str) -> str:
        return expression

    @tool(registry, "读文件")
    def read_file(path: str) -> str:
        return path

    # 超过阈值，只召回 top_n
    recalled = registry.recall("今天北京天气怎么样", top_n=1)
    assert len(recalled) == 1
    assert recalled[0].name == "get_weather"

    # 不超过阈值返回全部
    small = ToolRegistry(AgentConfig(tool_recall_threshold=8))
    register_fixture_tools(small)
    assert len(small.recall("任意", top_n=2)) == 4


def test_core_tools_exclude_fabricated_weather() -> None:
    registry = ToolRegistry()
    register_core_tools(registry)
    names = {t.name for t in registry.all()}
    assert names == {"calculator", "read_file", "get_current_time"}


def test_demo_get_weather_cities() -> None:
    registry = ToolRegistry()
    register_fixture_tools(registry)
    weather = registry.get("get_weather")
    assert "25℃" in weather.fn(city="北京")
    assert "18℃" in weather.fn(city="上海")
    assert "27℃" in weather.fn(city="广州")
    assert "26℃" in weather.fn(city="深圳")
    assert "22℃" in weather.fn(city="成都")
    assert "暂无数据" in weather.fn(city="火星")


def test_demo_calculator_ast_safe() -> None:
    registry = ToolRegistry()
    register_fixture_tools(registry)
    calc = registry.get("calculator")
    assert calc.fn(expression="25 - 18") == "7"
    assert calc.fn(expression="(1+2)*3") == "9"
    assert "失败" in calc.fn(expression="__import__('os')")


def test_demo_read_file_and_time(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_fixture_tools(registry)
    path = tmp_path / "note.txt"
    path.write_text("你好" * 2000, encoding="utf-8")
    content = registry.get("read_file").fn(path=str(path))
    assert len(content) == 2000
    assert "不存在" in registry.get("read_file").fn(path=str(tmp_path / "missing.txt"))
    now = registry.get("get_current_time").fn()
    assert len(now) >= 19
