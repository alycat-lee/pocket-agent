from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.report import render_radar, render_table, write_report
from benchmark.run import parse_dims
from benchmark.runners import TrackResult
from benchmark.schema import (
    DIMENSIONS,
    TASKS_DIR,
    TaskSchemaError,
    load_tasks,
    validate_task,
    validate_task_file,
)
from benchmark.score import JUDGE_PROMPT, mean_score, score_result


def test_all_dimension_task_files_match_schema() -> None:
    for dim in DIMENSIONS:
        matches = list(TASKS_DIR.glob(f"{dim}_*.json"))
        assert len(matches) == 1, dim
        tasks = validate_task_file(matches[0])
        assert len(tasks) >= 5
        assert all(t["dimension"] == dim for t in tasks)


def test_accident_sentence_in_d4() -> None:
    tasks = load_tasks(["d4"])
    inputs = [t["input"] for t in tasks]
    assert "给我生成一篇2万字的小说 内容随意" in inputs
    accident = next(t for t in tasks if "2万字" in t["input"])
    assert accident["expected_constraints"]["type"] == "writer"
    assert accident["expected_constraints"]["total_chars"] == 20000


def test_validate_task_rejects_bad_payload() -> None:
    with pytest.raises(TaskSchemaError):
        validate_task({"id": "x"})
    with pytest.raises(TaskSchemaError):
        validate_task(
            {
                "id": "x",
                "dimension": "d1",
                "input": "hi",
                "rubric": {"pass_criteria": [], "score_fields": []},
            }
        )


def test_scorer_deterministic_known_inputs() -> None:
    task = {
        "id": "t1",
        "dimension": "d1",
        "input": "json",
        "parse": {"kind": "json", "keys": ["name", "age"]},
        "expected": {"name": "Ada", "age": 30},
        "rubric": {"pass_criteria": ["json"], "score_fields": ["parse_ok"]},
    }
    ok = TrackResult(track="bare", task_id="t1", dimension="d1", answer='{"name":"Ada","age":30}')
    bad = TrackResult(track="bare", task_id="t1", dimension="d1", answer="not json")
    first = score_result(task, ok)
    second = score_result(task, ok)
    assert first["fields"] == second["fields"]
    assert first["fields"]["parse_ok"] == 1.0
    assert score_result(task, bad)["fields"]["parse_ok"] == 0.0


def test_scorer_needle_type_and_think() -> None:
    needle_task = {
        "id": "n",
        "dimension": "d2",
        "input": "x",
        "needle": "林小满",
        "rubric": {"pass_criteria": ["hit"], "score_fields": ["needle_hit"]},
    }
    hit = TrackResult(track="pocket", task_id="n", dimension="d2", answer="她叫林小满。")
    miss = TrackResult(track="pocket", task_id="n", dimension="d2", answer="不知道")
    assert score_result(needle_task, hit)["fields"]["needle_hit"] == 1.0
    assert score_result(needle_task, miss)["fields"]["needle_hit"] == 0.0

    plan_task = {
        "id": "p",
        "dimension": "d4",
        "input": "给我生成一篇2万字的小说 内容随意",
        "expected_constraints": {"type": "writer", "total_chars": 20000},
        "rubric": {
            "pass_criteria": ["type"],
            "score_fields": ["type_ok", "constraint_ok"],
        },
    }
    card_ok = TrackResult(
        track="pocket",
        task_id="p",
        dimension="d4",
        answer="goal",
        extras={
            "task_card": {
                "type": "writer",
                "goal": "写小说",
                "constraints": {"total_chars": 20000},
                "plan": [],
                "needs_history": False,
            }
        },
    )
    scored = score_result(plan_task, card_ok)
    assert scored["fields"]["type_ok"] == 1.0
    assert scored["fields"]["constraint_ok"] == 1.0

    think_task = {
        "id": "th",
        "dimension": "d5",
        "input": "q",
        "rubric": {"pass_criteria": ["no think"], "score_fields": ["no_think_leak"]},
    }
    leaked = TrackResult(track="bare", task_id="th", dimension="d5", answer="<think>x</think>是")
    clean = TrackResult(track="bare", task_id="th", dimension="d5", answer="北京是中国的首都。")
    assert score_result(think_task, leaked)["fields"]["no_think_leak"] == 0.0
    assert score_result(think_task, clean)["fields"]["no_think_leak"] == 1.0


def test_judge_prompt_is_fixed_1_to_5() -> None:
    assert "1 到 5" in JUDGE_PROMPT
    assert "{task_input}" in JUDGE_PROMPT
    assert "{criteria}" in JUDGE_PROMPT
    assert "只输出 JSON" in JUDGE_PROMPT


def test_report_format_and_delta(tmp_path: Path) -> None:
    scores = {
        "bare": {"d1": 0.40, "d2": 0.20, "d3": 0.00},
        "pocket": {"d1": 0.90, "d2": 0.80, "d3": 1.00},
        "cloud": {"d1": None, "d2": None, "d3": None},
    }
    table = render_table(scores, ["d1", "d2", "d3"])
    assert "bare" in table and "pocket" in table and "cloud" in table
    assert "D1" in table
    radar = render_radar(scores, ["d1", "d2", "d3"])
    assert "ASCII 雷达" in radar
    assert "pocket" in radar

    last = tmp_path / ".last_run.json"
    last.write_text(
        json.dumps({"scores": {"bare": {"d1": 0.30}, "pocket": {"d1": 0.70}, "cloud": {}}}),
        encoding="utf-8",
    )
    md_path = tmp_path / "benchmark_report.md"
    md = write_report(
        scores=scores,
        dims=["d1"],
        details=[],
        output=md_path,
        last_path=last,
        skipped={"cloud": "DEEPSEEK_API_KEY 缺失"},
        notes=[],
    )
    assert md_path.is_file()
    assert "与上次运行对比" in md
    assert "+0.10" in md or "+0.20" in md
    assert "DEEPSEEK_API_KEY" in md
    assert last.is_file()


def test_d2_d9_d3_tasks_encode_phase45_contracts() -> None:
    """维度回归：D2/D9 靠跨轮针，D3 含「现在几点了」工具题。"""
    d2 = {t["id"]: t for t in load_tasks(["d2"])}
    assert d2["d2-number"]["needle"] == "7381"
    d9 = {t["id"]: t for t in load_tasks(["d9"])}
    assert any("上海" in str(t) for t in d9["d9-weather-follow"]["turns"])
    d3 = {t["id"]: t for t in load_tasks(["d3"])}
    assert d3["d3-now"]["input"] == "现在几点了"


def test_scorer_forbidden_web_search_scores_zero_on_weather() -> None:
    task = {
        "id": "wx",
        "dimension": "d3",
        "input": "查北京天气",
        "expected_tools": ["get_weather"],
        "forbidden_tools": ["search_knowledge", "wiki_lookup", "web_search"],
        "rubric": {
            "pass_criteria": ["tool"],
            "score_fields": ["tool_used"],
        },
    }
    result = TrackResult(
        track="pocket",
        task_id="wx",
        dimension="d3",
        answer="网上说北京晴",
        extras={
            "real_calls": 1,
            "tool_calls": [
                {
                    "name": "web_search",
                    "args": {"query": "北京天气"},
                    "from_cache": False,
                    "error": None,
                }
            ],
        },
    )
    assert score_result(task, result)["fields"]["tool_used"] == 0.0


def test_d3_tasks_declare_expected_and_forbidden_tools() -> None:
    d3 = {t["id"]: t for t in load_tasks(["d3"])}
    for weather_id in (
        "d3-weather-bj",
        "d3-how-weather",
        "d3-two-city",
        "d3-three-city",
    ):
        assert d3[weather_id]["expected_tools"] == ["get_weather"]
        assert "wiki_lookup" in d3[weather_id]["forbidden_tools"]
    assert d3["d3-now"]["expected_tools"] == ["get_current_time"]


def test_dup_rate_ignores_fill_errors() -> None:
    from types import SimpleNamespace

    from benchmark.runners.pocket import _dup_call_rate

    steps = [
        SimpleNamespace(tool_name="get_weather", args={"city": "北京"}, error=None),
        SimpleNamespace(tool_name="get_weather", args={"city": "上海"}, error=None),
        SimpleNamespace(tool_name="get_weather", args={}, error="invalid json"),
        SimpleNamespace(tool_name="(format_error)", args={}, error="bad"),
    ]
    assert _dup_call_rate(2, steps) == 0.0
    assert _dup_call_rate(3, steps) == 1.0


def test_scorer_zero_dup_rate_is_success() -> None:
    task = {
        "id": "t",
        "dimension": "d3",
        "input": "查北京天气",
        "rubric": {
            "pass_criteria": ["tool"],
            "score_fields": ["tool_used", "dup_call_rate"],
        },
    }
    result = TrackResult(
        track="pocket",
        task_id="t",
        dimension="d3",
        answer="北京：晴，25℃",
        extras={"real_calls": 1, "dup_call_rate": 0.0},
    )
    fields = score_result(task, result)["fields"]
    assert fields["tool_used"] == 1.0
    assert fields["dup_call_rate"] == 1.0


def test_mean_score_ignores_skip() -> None:
    assert mean_score({"a": 1.0, "b": None}) == 1.0
    assert mean_score({"a": None}) is None


def test_parse_dims_rejects_unknown() -> None:
    assert parse_dims("d1,d2") == ["d1", "d2"]
    with pytest.raises(ValueError):
        parse_dims("d99")


def test_eval_help_lists_prompt_variant() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prompt-variant" in result.stdout
    assert "auto" in result.stdout


def test_bind_prompt_variant_auto_local_cloud() -> None:
    from benchmark.run import _bind_prompt_variant
    from pocket_agent.prompt_registry import current_registry
    from pocket_agent.prompts import CHOOSE_TOOL

    _bind_prompt_variant("cloud")
    assert current_registry() is not None
    assert current_registry().active_variant() == "cloud"
    cloud = CHOOSE_TOOL.format(task="t", state="s", history_text="（无）", options="0. x")
    assert "下一步应该使用哪个工具" in cloud

    _bind_prompt_variant("local")
    assert current_registry().active_variant() == "local"
    local = CHOOSE_TOOL.format(task="t", state="s", history_text="（无）", options="0. x")
    assert "选编号" in local

    _bind_prompt_variant("auto")
    assert current_registry().active_variant() == "local"
