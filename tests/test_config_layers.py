from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest
from pocket_agent.config import (
    AgentConfig,
    extract_base_from_pocket,
    load_settings,
    redact_secrets,
    save_settings,
)
from pocket_agent.config_schema import (
    assert_schema_covers_agent_config,
    get_config_schema,
    schema_keys,
)


def test_from_layers_priority() -> None:
    cfg = AgentConfig.from_layers(
        base={"temperature": 0.1, "max_steps": 5},
        settings={"temperature": 0.5, "max_tokens": 100},
        overrides={"temperature": 0.9},
    )
    assert cfg.temperature == 0.9
    assert cfg.max_steps == 5
    assert cfg.max_tokens == 100


def test_from_layers_ignores_unknown_keys() -> None:
    cfg = AgentConfig.from_layers(
        base={"temperature": 0.2, "not_a_field": 1},
        settings={"also_bad": True},
        overrides={"max_steps": 3},
    )
    assert cfg.temperature == 0.2
    assert cfg.max_steps == 3
    assert not hasattr(cfg, "not_a_field")


def test_overrides_do_not_pollute_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_settings({"temperature": 0.3}, path)
    before = load_settings(path)
    AgentConfig.from_layers(
        base=AgentConfig(),
        settings=before,
        overrides={"temperature": 0.99, "max_steps": 2},
    )
    after = load_settings(path)
    assert after == before
    assert after["temperature"] == 0.3
    assert "max_steps" not in after


def test_settings_atomic_write_and_corrupt_fallback(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    path = tmp_path / "settings.json"
    save_settings({"temperature": 0.4, "api_key": "sk-secret"}, path)
    assert path.is_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["temperature"] == 0.4

    path.write_text("{not-json", encoding="utf-8")
    data = load_settings(path)
    assert data == {}
    assert any("corrupt" in str(w.message).lower() for w in recwarn)


def test_redact_api_key() -> None:
    out = redact_secrets({"api_key": "sk-abcdefgh", "temperature": 0.2})
    assert out["api_key"] == "sk-***"
    assert out["temperature"] == 0.2
    assert redact_secrets({"api_key": "ab"})["api_key"] == "***"
    assert redact_secrets({"api_key": ""})["api_key"] == ""


def test_schema_covers_all_agent_config_fields() -> None:
    assert_schema_covers_agent_config()
    assert schema_keys() == {f.name for f in fields(AgentConfig)}
    visible = {item["key"] for item in get_config_schema()}
    assert "permission_rules" not in visible


def test_settings_permission_rules_override_base_not_request() -> None:
    settings_rules = [{"effect": "allow", "tool": "run_command", "pattern": "^echo "}]
    cfg = AgentConfig.from_layers(
        base={"permission_rules": [{"effect": "deny", "tool": "*"}]},
        settings={"permission_rules": settings_rules},
        overrides={"permission_rules": [{"effect": "allow", "tool": "*"}]},
    )
    assert cfg.permission_rules == settings_rules


def test_extract_base_from_pocket() -> None:
    pocket = {
        "plugins": [
            {"name": "llm-llamacpp", "config": {"base_url": "http://x/v1", "temperature": 0.1}},
            {"name": "agent-loop", "config": {"max_steps": 7}},
            {"name": "other", "disabled": True, "config": {"max_steps": 99}},
        ]
    }
    base = extract_base_from_pocket(pocket)
    assert base["base_url"] == "http://x/v1"
    assert base["temperature"] == 0.1
    assert base["max_steps"] == 7
