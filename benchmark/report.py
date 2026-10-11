"""维度得分表 + 终端 ASCII 雷达图 + benchmark_report.md（含上次差值）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.schema import DIM_LABELS, DIMENSIONS, TRACKS

LAST_RUN_NAME = ".last_run.json"


def render_table(scores: dict[str, dict[str, float | None]], dims: list[str]) -> str:
    headers = ["维度", *TRACKS]
    widths = [18, 8, 8, 8]
    lines = [
        _row(headers, widths),
        _row(["-" * w for w in widths], widths, sep="-+-"),
    ]
    for dim in dims:
        label = DIM_LABELS.get(dim, dim)
        cells = [label[:18]]
        for track in TRACKS:
            cells.append(_fmt(scores.get(track, {}).get(dim)))
        lines.append(_row(cells, widths))
    return "\n".join(lines)


def render_radar(scores: dict[str, dict[str, float | None]], dims: list[str]) -> str:
    """每维三条进度条，构成终端雷达。"""
    lines = ["ASCII 雷达（0 -------- 1）"]
    for dim in dims:
        label = DIM_LABELS.get(dim, dim)
        lines.append(f"  {label}")
        for track in TRACKS:
            val = scores.get(track, {}).get(dim)
            lines.append(f"    {track:<6} {_bar(val)} {_fmt(val)}")
    return "\n".join(lines)


def write_report(
    *,
    scores: dict[str, dict[str, float | None]],
    dims: list[str],
    details: list[dict[str, Any]],
    output: Path,
    last_path: Path,
    skipped: dict[str, str],
    notes: list[str],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    previous = _load_last(last_path)
    table = render_table(scores, dims)
    radar = render_radar(scores, dims)
    delta = _delta_table(scores, previous, dims)
    md = "\n".join(
        [
            "# 能力等效基准报告",
            "",
            f"- 时间：{now}",
            f"- 维度：{', '.join(dims)}",
            f"- 跳过：{skipped or '无'}",
            *[f"- {n}" for n in notes],
            "",
            "## 维度得分（三跑道并列）",
            "",
            "```",
            table,
            "```",
            "",
            "## 与上次运行对比",
            "",
            delta,
            "",
            "## 雷达图",
            "",
            "```",
            radar,
            "```",
            "",
            "## 明细",
            "",
            "```json",
            json.dumps(details, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    output.write_text(md, encoding="utf-8")
    last_path.write_text(
        json.dumps({"ts": now, "dims": dims, "scores": scores}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md


def _row(cells: list[str], widths: list[int], *, sep: str = " | ") -> str:
    return sep.join(c[:w].ljust(w) for c, w in zip(cells, widths, strict=True))


def _fmt(value: float | None) -> str:
    if value is None:
        return "skip"
    return f"{value:.2f}"


def _bar(value: float | None, width: int = 10) -> str:
    if value is None:
        return "----------"
    filled = max(0, min(width, round(value * width)))
    return "#" * filled + "-" * (width - filled)


def _load_last(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _delta_table(
    scores: dict[str, dict[str, float | None]],
    previous: dict[str, Any],
    dims: list[str],
) -> str:
    old = previous.get("scores") if isinstance(previous.get("scores"), dict) else {}
    if not old:
        return "无上次记录，本次作为基线。"
    lines = [
        "| 维度 | 跑道 | 本次 | 上次 | Δ |",
        "|---|---|---|---|---|",
    ]
    for dim in dims:
        for track in TRACKS:
            cur = scores.get(track, {}).get(dim)
            prev = None
            track_old = old.get(track)
            if isinstance(track_old, dict):
                raw = track_old.get(dim)
                if isinstance(raw, (int, float)):
                    prev = float(raw)
            if cur is None and prev is None:
                delta = "skip"
            elif cur is None or prev is None:
                delta = "n/a"
            else:
                delta = f"{cur - prev:+.2f}"
            lines.append(
                f"| {DIM_LABELS.get(dim, dim)} | {track} | {_fmt(cur)} | {_fmt(prev)} | {delta} |"
            )
    return "\n".join(lines)


def empty_scores() -> dict[str, dict[str, float | None]]:
    return {track: {dim: None for dim in DIMENSIONS} for track in TRACKS}
