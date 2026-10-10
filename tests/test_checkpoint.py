from __future__ import annotations

from pathlib import Path

import pytest
from pocket_agent.checkpoint import CheckpointStore
from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMConnectionError
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry, tool


def _make_runner(
    mock_llm_factory: type,
    responses: list[object],
    calls: dict[str, int],
    *,
    max_steps: int = 12,
) -> AgentRunner:
    config = AgentConfig(max_steps=max_steps, tool_recall_top_n=5)
    registry = ToolRegistry(config)

    @tool(registry, "记录一个整数值")
    def record_value(v: int) -> str:
        calls["n"] += 1
        return f"ok:{v}"

    llm = mock_llm_factory(responses)
    return AgentRunner(config, llm, registry)


def test_resume_no_duplicate(
    mock_llm_factory: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}
    run_id = "resume-demo"

    runner1 = _make_runner(
        mock_llm_factory,
        [
            0,
            {"v": 1},
            0,
            {"v": 2},
            0,
            {"v": 3},
            0,
            {"v": 4},
            LLMConnectionError("模拟断连"),
        ],
        calls,
    )
    with pytest.raises(LLMConnectionError, match="模拟断连"):
        runner1.run("依次记录数值 1 到 5", run_id=run_id)

    assert calls["n"] == 4
    assert CheckpointStore().load(run_id) is not None
    real_after_partial = runner1._ledger.real_call_count
    assert real_after_partial == 4

    runner2 = _make_runner(
        mock_llm_factory,
        [-1, "四步均已完成"],
        calls,
    )
    result = runner2.run("依次记录数值 1 到 5", run_id=run_id)

    assert calls["n"] == 4
    assert result.stats["real_calls"] == 0
    assert all(s.from_cache for s in result.steps[:4])
    assert len(result.steps) == 4
    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.answer == "四步均已完成"


def test_checkpoint_cleared(
    mock_llm_factory: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}
    run_id = "clear-demo"
    runner = _make_runner(
        mock_llm_factory,
        [0, {"v": 1}, -1, "完成"],
        calls,
    )
    result = runner.run("记录数值 1", run_id=run_id)
    assert result.stop_reason == StopReason.MODEL_DONE
    assert CheckpointStore().load(run_id) is None
    assert not (tmp_path / ".checkpoints" / "clear-demo.json").exists()


def test_corrupt_checkpoint(
    mock_llm_factory: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}
    run_id = "corrupt-demo"
    store = CheckpointStore()
    path = tmp_path / ".checkpoints" / "corrupt-demo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json!!!", encoding="utf-8")
    assert store.load(run_id) is None

    runner = _make_runner(
        mock_llm_factory,
        [0, {"v": 9}, -1, "从头完成"],
        calls,
    )
    result = runner.run("记录数值 9", run_id=run_id)
    assert result.stop_reason == StopReason.MODEL_DONE
    assert result.answer == "从头完成"
    assert calls["n"] == 1
    assert CheckpointStore().load(run_id) is None
