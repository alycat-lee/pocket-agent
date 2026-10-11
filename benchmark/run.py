"""编排三跑道：跑任务 → 评分 → 写报告。"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pocket_agent.llm_client import LLMClientProtocol
from pocket_agent.plugins.prompts_plugin import PromptsPlugin
from pocket_agent.prompt_registry import current_registry
from pocket_agent.runtime import Context

from benchmark.report import LAST_RUN_NAME, empty_scores, render_radar, render_table, write_report
from benchmark.runners.bare import BareRunner
from benchmark.runners.cloud import CloudRunner, cloud_available, make_judge_client
from benchmark.runners.pocket import PocketRunner
from benchmark.schema import DIMENSIONS, TASKS_DIR, TRACKS, load_tasks
from benchmark.score import mean_score, score_result

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_dims(raw: str) -> list[str]:
    dims = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    unknown = [d for d in dims if d not in DIMENSIONS]
    if unknown:
        raise ValueError(f"unknown dimensions: {unknown}")
    return dims or ["d1", "d2", "d3"]


def parse_tracks(raw: str) -> list[str]:
    tracks = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    unknown = [t for t in tracks if t not in TRACKS]
    if unknown:
        raise ValueError(f"unknown tracks: {unknown}")
    return tracks or list(TRACKS)


def run_benchmark(
    *,
    dims: list[str],
    tracks: list[str],
    base_url: str,
    output: Path,
    tasks_dir: Path | None = None,
    judge: bool = False,
    prompt_variant: str = "auto",
) -> int:
    tasks = load_tasks(dims, tasks_dir=tasks_dir or TASKS_DIR)
    _bind_prompt_variant(prompt_variant)
    runners = _make_runners(tracks, base_url)
    skipped: dict[str, str] = {}
    if "cloud" in tracks and not cloud_available():
        skipped["cloud"] = "DEEPSEEK_API_KEY 缺失"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output.parent / "benchmark_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = out_dir / "judge"
    judge_client: LLMClientProtocol | None = make_judge_client() if judge else None

    details: list[dict[str, Any]] = []
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for task in tasks:
        for name in tracks:
            runner = runners[name]
            print(f"==> {name} {task['id']}", flush=True)
            result = runner.run_task(task)
            if result.skipped:
                skipped.setdefault(name, result.skip_reason)
                print(f"    skip {result.skip_reason}", flush=True)
            elif result.error:
                print(f"    error {result.error}", flush=True)
            scored = score_result(
                task,
                result,
                judge=judge_client,
                judge_dir=judge_dir if judge_client else None,
            )
            avg = mean_score(scored["fields"])
            if avg is not None:
                bucket[name][str(task["dimension"])].append(avg)
            row = {
                "track": name,
                "task_id": task["id"],
                "dimension": task["dimension"],
                "answer": result.answer,
                "error": result.error,
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
                "extras": result.extras,
                "scores": scored["fields"],
                "mean": avg,
                "judge": scored["judge"],
            }
            details.append(row)
            print(f"    mean={_fmt(avg)} fields={scored['fields']}", flush=True)

    scores = empty_scores()
    for track in tracks:
        for dim in dims:
            vals = bucket[track][dim]
            scores[track][dim] = sum(vals) / len(vals) if vals else None
        if track in skipped and all(
            r.get("skipped") for r in details if r["track"] == track
        ):
            for dim in dims:
                scores[track][dim] = None

    last_path = REPO_ROOT / "benchmark" / LAST_RUN_NAME
    notes = [
        f"明细 JSON：{out_dir / 'details.json'}",
        f"prompt_variant={prompt_variant}",
    ]
    if skipped:
        notes.append("cloud 跳过时维度记 skip，不参与对比分母")
    write_report(
        scores=scores,
        dims=dims,
        details=details,
        output=output,
        last_path=last_path,
        skipped=skipped,
        notes=notes,
    )
    (out_dir / "details.json").write_text(
        json.dumps({"scores": scores, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(render_table(scores, dims))
    print()
    print(render_radar(scores, dims))
    print(f"\nreport: {output}")
    return 0


def _bind_prompt_variant(prompt_variant: str) -> None:
    """为 pocket 跑道挂上内置提示词，并按 --prompt-variant 锁定 A/B。"""
    ctx = Context()
    ctx.mount(PromptsPlugin())
    registry = current_registry()
    if registry is None:
        return
    if prompt_variant in {"local", "cloud"}:
        registry.set_variant(prompt_variant)  # type: ignore[arg-type]
    else:
        registry.select_for_provider("llamacpp")


def _make_runners(tracks: list[str], base_url: str) -> dict[str, Any]:
    runners: dict[str, Any] = {}
    if "bare" in tracks:
        runners["bare"] = BareRunner(base_url)
    if "pocket" in tracks:
        runners["pocket"] = PocketRunner(base_url)
    if "cloud" in tracks:
        runners["cloud"] = CloudRunner()
    return runners


def _fmt(value: float | None) -> str:
    return "skip" if value is None else f"{value:.2f}"
