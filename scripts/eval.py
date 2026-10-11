#!/usr/bin/env python3
"""pocket-agent 真机评测：框架模式 vs 裸自由 Agent 基线。"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pocket_agent.config import AgentConfig
from pocket_agent.ledger import CallLedger, canonical_json
from pocket_agent.llm_client import LlamaCppClient, LLMConnectionError
from pocket_agent.runner import AgentRunner, StopReason
from pocket_agent.tools import ToolRegistry
from pocket_agent.writer import LongWriter, WriterMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_TESTS = REPO_ROOT / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from benchmark.run import parse_dims, parse_tracks, run_benchmark
from benchmark.schema import TRACKS

from fixtures.fixture_tools import register_fixture_tools

REPORT_PATH = REPO_ROOT / "tests" / "fixtures" / "report.txt"
DEFAULT_BASE_URL = AgentConfig().base_url

TASKS: list[dict[str, str]] = [
    {"id": "one_step", "task": "现在几点了"},
    {"id": "two_city", "task": "用计算器算 25 减 18，再算 100 除以 4"},
    {"id": "three_city", "task": "用计算器分别算 (1+2)*3 和 10-4"},
    {
        "id": "long_file",
        "task": f"读取 {REPORT_PATH.as_posix()} 并总结要点",
    },
    {"id": "many_tools", "task": "现在几点了"},
]


def _make_registry(config: AgentConfig, *, many_tools: bool) -> ToolRegistry:
    registry = ToolRegistry(config)
    register_fixture_tools(registry)
    if many_tools:
        for i in range(11):
            idx = i

            def _fn(query: str = "", *, _idx: int = idx) -> str:
                return f"placeholder-{_idx}:{query}"

            _fn.__name__ = f"placeholder_{idx}"
            registry.register(_fn, short_desc=f"占位工具{idx}")
    return registry


def _unique_fps(steps: list[Any]) -> set[str]:
    ledger = CallLedger()
    return {
        ledger.fingerprint(s.tool_name, s.args)
        for s in steps
        if getattr(s, "tool_name", "") != "(format_error)"
    }


def run_pocket_once(task_id: str, task: str, *, base_url: str) -> dict[str, Any]:
    config = AgentConfig(base_url=base_url)
    registry = _make_registry(config, many_tools=(task_id == "many_tools"))
    runner = AgentRunner(config, LlamaCppClient(config), registry)
    result = runner.run(task)
    fps = _unique_fps(result.steps)
    real_calls = result.stats["real_calls"]
    dup_rate = 0.0 if real_calls == len(fps) else 1.0
    success = (
        result.stop_reason == StopReason.MODEL_DONE
        and dup_rate == 0.0
        and result.stats["prompt_tokens_max"] <= 3500
        and bool(result.answer.strip())
    )
    return {
        "success": success,
        "dup_call_rate": dup_rate,
        "intercepts": result.stats["intercepts"],
        "steps": result.stats["total_steps"],
        "prompt_tokens": result.stats["prompt_tokens_max"],
        "real_calls": real_calls,
        "stop_reason": result.stop_reason.value,
        "answer": result.answer,
    }


def _chat(base_url: str, messages: list[dict[str, str]], *, timeout: float = 60.0) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def _demo_tool_fns() -> dict[str, Callable[..., str]]:
    """裸循环直接调用的工具函数表（复用测试道具注册结果）。"""
    registry = ToolRegistry(AgentConfig())
    register_fixture_tools(registry)
    return {t.name: t.fn for t in registry.all()}


def run_baseline_once(task: str, *, base_url: str) -> dict[str, Any]:
    """裸自由 Agent Loop（约 30 行）：无账本/压缩/护栏，重复也会真实调用。"""
    tools = _demo_tool_fns()
    tool_names = ", ".join(tools)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                f"可用工具: {tool_names}。"
                '调用格式: CALL name {"arg":"v"}；结束: FINAL 回答。不要解释。'
            ),
        },
        {"role": "user", "content": task},
    ]
    real_calls = 0
    dup_calls = 0
    seen: set[str] = set()
    answer = ""
    for _ in range(8):
        content = _chat(base_url, messages)
        messages.append({"role": "assistant", "content": content})
        if "FINAL" in content:
            answer = content.split("FINAL", 1)[-1].strip(" :\n")
            break
        match = re.search(r"CALL\s+(\w+)\s+(\{.*\})", content, flags=re.DOTALL)
        if not match:
            messages.append({"role": "user", "content": "请用 CALL 或 FINAL 格式。"})
            continue
        name, raw_args = match.group(1), match.group(2)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "参数 JSON 无效，请重试。"})
            continue
        if name not in tools:
            messages.append({"role": "user", "content": f"未知工具 {name}"})
            continue
        fp = name + canonical_json(args)
        if fp in seen:
            dup_calls += 1
        seen.add(fp)
        real_calls += 1
        result_text = tools[name](**args)
        messages.append({"role": "user", "content": f"工具结果: {result_text}"})
    else:
        answer = messages[-1]["content"] if messages else ""
    dup_rate = (dup_calls / real_calls) if real_calls else 0.0
    success = bool(answer.strip()) and real_calls >= 1
    return {
        "success": success,
        "dup_call_rate": dup_rate,
        "intercepts": 0,
        "steps": real_calls,
        "prompt_tokens": 0,
        "real_calls": real_calls,
        "stop_reason": "baseline_done",
        "answer": answer,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "success_rate": 0.0,
            "dup_call_rate": 0.0,
            "avg_intercepts": 0.0,
            "avg_steps": 0.0,
            "avg_prompt_tokens": 0.0,
        }
    return {
        "success_rate": sum(1 for r in rows if r["success"]) / len(rows),
        "dup_call_rate": statistics.fmean(r["dup_call_rate"] for r in rows),
        "avg_intercepts": statistics.fmean(r["intercepts"] for r in rows),
        "avg_steps": statistics.fmean(r["steps"] for r in rows),
        "avg_prompt_tokens": statistics.fmean(r["prompt_tokens"] for r in rows),
    }


def _print_table(title: str, by_task: dict[str, dict[str, float]]) -> None:
    print()
    print(title)
    headers = [
        "task",
        "success_rate",
        "dup_call_rate",
        "avg_intercepts",
        "avg_steps",
        "avg_prompt_tokens",
    ]
    print(" | ".join(f"{h:>16}" for h in headers))
    print("-+-".join("-" * 16 for _ in headers))
    for task_id, metrics in by_task.items():
        row = [
            f"{task_id:>16}",
            f"{metrics['success_rate']:>16.2%}",
            f"{metrics['dup_call_rate']:>16.2%}",
            f"{metrics['avg_intercepts']:>16.2f}",
            f"{metrics['avg_steps']:>16.2f}",
            f"{metrics['avg_prompt_tokens']:>16.1f}",
        ]
        print(" | ".join(row))


def run_writer_eval(*, base_url: str, runs: int, output: Path) -> int:
    """长文本评测：3000 字 × runs，检查段间重叠 / 数字一致性 / 大纲覆盖。"""
    topic = "新能源汽车行业分析"
    material = ""
    if REPORT_PATH.is_file():
        material = REPORT_PATH.read_text(encoding="utf-8")[:2000]

    section_target = AgentConfig().section_char_target
    config = AgentConfig(
        base_url=base_url,
        max_tokens=max(1536, section_target * 3),
        timeout=300.0,
    )
    llm = LlamaCppClient(config)
    metrics = WriterMetrics()
    rows: list[dict[str, Any]] = []

    for i in range(runs):
        print(f"==> writer run {i + 1}/{runs}")
        writer = LongWriter(config, llm)
        try:
            article = writer.write(topic, total_chars=3000, material=material)
        except LLMConnectionError as exc:
            print(f"连接失败: {exc}", file=sys.stderr)
            return 2

        sections = writer.last_sections
        window = config.overlap_max
        pair_overlaps = [
            metrics.seam_ngram_overlap(a, b, n=3, window=window)
            for a, b in itertools.pairwise(sections)
        ]
        max_pair = max(pair_overlaps) if pair_overlaps else 0.0
        conflicts = metrics.numeric_consistency(article)
        coverage = metrics.outline_coverage(writer.last_outline, article)
        char_ok = len(article) >= int(3000 * 0.85)
        ok = (
            max_pair < 0.05
            and len(conflicts) == 0
            and coverage >= 0.90
            and char_ok
        )
        row = {
            "success": ok,
            "max_ngram_overlap": max_pair,
            "numeric_conflicts": conflicts,
            "outline_coverage": coverage,
            "chars": len(article),
            "sections": len(sections),
            "fallback": writer.used_outline_fallback,
        }
        rows.append(row)
        print(
            f"  overlap={max_pair:.2%} conflicts={len(conflicts)} "
            f"coverage={coverage:.2%} chars={len(article)} ok={ok}"
        )

        # 人工抽检：打印前两个接缝附近
        if i == 0 and len(sections) >= 2:
            print("\n===== 人工抽检接缝 1 =====")
            print(sections[0][-120:])
            print("---")
            print(sections[1][:120])
        if i == 0 and len(sections) >= 3:
            print("\n===== 人工抽检接缝 2 =====")
            print(sections[1][-120:])
            print("---")
            print(sections[2][:120])

    success_rate = sum(1 for r in rows if r["success"]) / len(rows) if rows else 0.0
    payload = {
        "mode": "writer",
        "base_url": base_url,
        "runs": runs,
        "success_rate": success_rate,
        "raw": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwriter success_rate={success_rate:.2%} saved: {output}")
    if success_rate < 1.0:
        print("WARNING: 部分 writer 评测未达标", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pocket-agent 真机评测")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="重复次数（框架评测默认 10；--writer 默认 5）",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="同时跑裸自由 Agent 基线并输出对比表",
    )
    parser.add_argument(
        "--writer",
        action="store_true",
        help="长文本评测模式：3000 字 × runs，检查连贯性指标",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "eval_results.json",
        help="结果 JSON 路径",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="三跑道能力基准（文档 15）",
    )
    parser.add_argument(
        "--dims",
        default="d1,d2,d3",
        help="基准维度，逗号分隔，如 d1,d2,d3",
    )
    parser.add_argument(
        "--tracks",
        default="bare,pocket,cloud",
        help="基准跑道，逗号分隔",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="对 rubric 主观项启用云端 judge（需 DEEPSEEK_API_KEY）",
    )
    args = parser.parse_args(argv)

    if args.benchmark:
        return _run_benchmark(args)

    probe_url = f"{args.base_url.rstrip('/')}/chat/completions"
    try:
        with httpx.Client(timeout=3.0) as client:
            probe = client.post(
                probe_url,
                json={
                    "model": "local-model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
        if probe.status_code != 200:
            print(
                f"llama-server 不可用（{probe_url} -> HTTP {probe.status_code}）。"
                "请先启动：llama-server -c 8192 --flash-attn --cache-reuse 256 --port 8000",
                file=sys.stderr,
            )
            return 2
    except httpx.HTTPError as exc:
        print(f"连接 LLM 失败：{exc}", file=sys.stderr)
        return 2

    if args.writer:
        return run_writer_eval(
            base_url=args.base_url,
            runs=args.runs if args.runs is not None else 5,
            output=args.output,
        )

    runs = args.runs if args.runs is not None else 10
    pocket_raw: dict[str, list[dict[str, Any]]] = {t["id"]: [] for t in TASKS}
    baseline_raw: dict[str, list[dict[str, Any]]] = {t["id"]: [] for t in TASKS}

    for meta in TASKS:
        task_id, task = meta["id"], meta["task"]
        print(f"==> pocket-agent: {task_id} x{runs}")
        for i in range(runs):
            try:
                row = run_pocket_once(task_id, task, base_url=args.base_url)
            except LLMConnectionError as exc:
                print(f"连接失败: {exc}", file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001
                row = {
                    "success": False,
                    "dup_call_rate": 1.0,
                    "intercepts": 0,
                    "steps": 0,
                    "prompt_tokens": 0,
                    "real_calls": 0,
                    "stop_reason": "error",
                    "answer": str(exc),
                }
            pocket_raw[task_id].append(row)
            print(
                f"  [{i + 1}/{runs}] success={row['success']} "
                f"dup={row['dup_call_rate']:.0%} steps={row['steps']}"
            )

        if args.baseline:
            print(f"==> baseline: {task_id} x{runs}")
            for i in range(runs):
                try:
                    row = run_baseline_once(task, base_url=args.base_url)
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "success": False,
                        "dup_call_rate": 1.0,
                        "intercepts": 0,
                        "steps": 0,
                        "prompt_tokens": 0,
                        "real_calls": 0,
                        "stop_reason": "error",
                        "answer": str(exc),
                    }
                baseline_raw[task_id].append(row)
                print(
                    f"  [{i + 1}/{runs}] success={row['success']} "
                    f"dup={row['dup_call_rate']:.0%} steps={row['steps']}"
                )

    pocket_metrics = {k: _aggregate(v) for k, v in pocket_raw.items()}
    _print_table("pocket-agent metrics", pocket_metrics)

    payload: dict[str, Any] = {
        "base_url": args.base_url,
        "runs": runs,
        "pocket_agent": {"by_task": pocket_metrics, "raw": pocket_raw},
    }

    if args.baseline:
        baseline_metrics = {k: _aggregate(v) for k, v in baseline_raw.items()}
        _print_table("baseline (naked loop) metrics", baseline_metrics)
        print()
        print("comparison (pocket - baseline):")
        for task_id, p in pocket_metrics.items():
            b = baseline_metrics[task_id]
            print(
                f"  {task_id}: success_delta="
                f"{p['success_rate'] - b['success_rate']:+.2%} "
                f"dup_delta={p['dup_call_rate'] - b['dup_call_rate']:+.2%}"
            )
        payload["baseline"] = {"by_task": baseline_metrics, "raw": baseline_raw}

    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nresults saved: {args.output}")

    # 框架模式要求平均重复调用率为 0
    max_dup = max(m["dup_call_rate"] for m in pocket_metrics.values())
    if max_dup > 0:
        print(f"WARNING: pocket-agent dup_call_rate > 0 (max={max_dup:.2%})", file=sys.stderr)
        return 1
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    try:
        dims = parse_dims(str(args.dims))
        tracks = parse_tracks(str(args.tracks))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    need_local = any(t in tracks for t in ("bare", "pocket"))
    if need_local:
        probe_url = f"{args.base_url.rstrip('/')}/chat/completions"
        try:
            with httpx.Client(timeout=3.0, trust_env=False) as client:
                probe = client.post(
                    probe_url,
                    json={
                        "model": "local-model",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    },
                )
            if probe.status_code != 200:
                print(
                    f"llama-server 不可用（{probe_url} -> HTTP {probe.status_code}）。",
                    file=sys.stderr,
                )
                return 2
        except httpx.HTTPError as exc:
            print(f"连接 LLM 失败：{exc}", file=sys.stderr)
            return 2

    output = args.output
    if output == REPO_ROOT / "eval_results.json":
        output = REPO_ROOT / "benchmark_report.md"
    return run_benchmark(
        dims=dims,
        tracks=tracks or list(TRACKS),
        base_url=str(args.base_url),
        output=output,
        judge=bool(args.judge),
    )


if __name__ == "__main__":
    raise SystemExit(main())
