from __future__ import annotations

from pathlib import Path
from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.permissions import builtin_rules, evaluate
from pocket_agent.plugins.workspace_plugin import fenced_write_file
from pocket_agent.runner import AgentRunner
from pocket_agent.tools import ToolRegistry
from pocket_agent.workspace import WorkspaceManager


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))


class _FakeLlm:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)

    def generate(self, prompt: str) -> str:
        return str(self._script.pop(0) if self._script else "")

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return int(self._script.pop(0) if self._script else -1)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        item = self._script.pop(0) if self._script else {}
        assert isinstance(item, dict)
        return item


def test_evaluate_order_short_circuit() -> None:
    rules = [
        {"effect": "allow", "tool": "run_command", "pattern": "echo"},
        {"effect": "deny", "tool": "run_command", "note": "其余命令拒绝"},
    ]
    allow, _ = evaluate(rules, "run_command", {"command": "echo hi"}, cacheable=False)
    deny, rule = evaluate(rules, "run_command", {"command": "ls"}, cacheable=False)
    assert allow == "allow"
    assert deny == "deny"
    assert rule.note == "其余命令拒绝"


def test_evaluate_tool_glob() -> None:
    rules = [{"effect": "deny", "tool": "write_*", "note": "禁止写"}]
    verdict, rule = evaluate(
        rules, "write_file", {"path": "a.txt", "content": "x"}, cacheable=False
    )
    miss, _ = evaluate(rules, "read_file", {"path": "a.txt"}, cacheable=True)
    assert verdict == "deny"
    assert rule.note == "禁止写"
    assert miss == "allow"


def test_evaluate_pattern_regex_on_dumps() -> None:
    rules = [{"effect": "deny", "tool": "write_file", "pattern": r'"path": "secret'}]
    hit, _ = evaluate(
        rules,
        "write_file",
        {"path": "secret.txt", "content": "x"},
        cacheable=False,
    )
    miss, _ = evaluate(
        rules,
        "write_file",
        {"path": "ok.txt", "content": "x"},
        cacheable=False,
    )
    assert hit == "deny"
    assert miss == "ask"


def test_evaluate_fallback_without_hit() -> None:
    allow, allow_rule = evaluate([], "calculator", {"expression": "1"}, cacheable=True)
    ask, ask_rule = evaluate([], "custom_write", {"x": "1"}, cacheable=False)
    assert allow == "allow"
    assert allow_rule.effect == "allow"
    assert ask == "ask"
    assert ask_rule.effect == "ask"


def test_user_rule_beats_builtin_deny() -> None:
    builtin_deny, builtin_rule = evaluate(
        [], "run_command", {"command": "rm -rf /"}, cacheable=False
    )
    assert builtin_deny == "deny"
    assert "递归" in (builtin_rule.note or "")
    assert any(r.pattern == r"rm\s+-[a-z]*r[a-z]*f" for r in builtin_rules())

    allow, _ = evaluate(
        [{"effect": "allow", "tool": "run_command", "pattern": "rm"}],
        "run_command",
        {"command": "rm -rf /"},
        cacheable=False,
    )
    assert allow == "allow"


def test_overrides_cannot_inject_permission_rules() -> None:
    injected = [{"effect": "allow", "tool": "*", "note": "越权"}]
    settings_rules = [{"effect": "deny", "tool": "run_command", "note": "settings"}]
    cfg = AgentConfig.from_layers(
        base={"permission_rules": [{"effect": "ask", "tool": "*"}]},
        settings={"permission_rules": settings_rules},
        overrides={"permission_rules": injected, "max_steps": 3},
    )
    assert cfg.max_steps == 3
    assert cfg.permission_rules == settings_rules
    empty = AgentConfig.from_layers(overrides={"permission_rules": injected})
    assert empty.permission_rules == []


def _write_registry(tmp_path: Path) -> tuple[ToolRegistry, Path]:
    settings = tmp_path / "settings.json"
    manager = WorkspaceManager(settings_path=settings)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    manager._workspaces["default"].root = str(sandbox)
    registry = ToolRegistry(AgentConfig(max_steps=5, tool_recall_threshold=20))

    def write_file(path: str, content: str) -> str:
        return fenced_write_file(manager, "s1", path, content)

    registry.register(write_file, "写入工作区内文件", cacheable=False)
    return registry, sandbox


def test_runner_deny_does_not_create_file(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    bus = _Bus()
    config = AgentConfig(
        max_steps=4,
        tool_recall_threshold=20,
        permission_rules=[{"effect": "deny", "tool": "write_file", "note": "禁止写盘"}],
    )
    runner = AgentRunner(
        config,
        _FakeLlm([0, {"path": "x.txt", "content": "no"}, -1, "done"]),
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: True
    result = runner.run("写文件", permission="workspace")
    assert not (sandbox / "x.txt").exists()
    assert any(
        payload.get("reason") == "permission_rule" for _n, payload in bus.events
    )
    assert all(name != "approval.required" for name, _ in bus.events)
    assert any(step.error and "禁止写盘" in step.error for step in result.steps)


def test_runner_ask_still_requires_approval(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    bus = _Bus()
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20),
        _FakeLlm([0, {"path": "y.txt", "content": "ask"}, -1, "done"]),
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: False
    result = runner.run("写文件", permission="workspace")
    assert not (sandbox / "y.txt").exists()
    assert any(name == "approval.required" for name, _ in bus.events)
    assert any(step.error and "拒绝" in (step.error or "") for step in result.steps)


def test_runner_allow_whitelist_skips_approval(tmp_path: Path) -> None:
    registry = ToolRegistry(AgentConfig(max_steps=4, tool_recall_threshold=20))
    calls: list[str] = []

    def run_command(command: str) -> str:
        calls.append(command)
        return f"退出码 0\n{command}"

    registry.register(run_command, "在工作区执行命令", cacheable=False)
    bus = _Bus()
    config = AgentConfig(
        max_steps=4,
        tool_recall_threshold=20,
        permission_rules=[
            {
                "effect": "allow",
                "tool": "run_command",
                "pattern": "^echo ",
                "note": "echo 白名单",
            }
        ],
    )
    runner = AgentRunner(
        config,
        _FakeLlm([0, {"command": "echo hi"}, -1, "done"]),
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: False
    result = runner.run("运行 echo hi", permission="workspace")
    assert calls == ["echo hi"]
    assert all(name != "approval.required" for name, _ in bus.events)
    assert any("echo hi" in step.result for step in result.steps)


def test_runner_no_rules_matches_legacy_non_cacheable_ask(tmp_path: Path) -> None:
    registry, sandbox = _write_registry(tmp_path)
    bus = _Bus()
    runner = AgentRunner(
        AgentConfig(max_steps=4, tool_recall_threshold=20, permission_rules=[]),
        _FakeLlm([0, {"path": "z.txt", "content": "legacy"}, -1, "done"]),
        registry,
        events=bus,
    )
    runner.approval_gate = lambda _payload: True
    runner.run("写文件", permission="workspace")
    assert (sandbox / "z.txt").exists()
    assert any(name == "approval.required" for name, _ in bus.events)
