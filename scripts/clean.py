#!/usr/bin/env python3
"""清除缓存与测试生成的临时文件（仅标准库，无第三方依赖）。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".pytest_tmp",
    ".ruff_cache",
    ".mypy_cache",
    "htmlcov",
    "dist",
    "build",
    ".eggs",
}

FILE_NAMES = {
    ".coverage",
}

FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".coverage",
}


def _rm_path(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)
        return False


def clean(root: Path = ROOT, *, dry_run: bool = False) -> int:
    removed = 0
    candidates: list[Path] = []

    for path in root.rglob("*"):
        if not path.exists():
            continue
        # 跳过虚拟环境，避免误删
        parts = set(path.parts)
        if ".venv" in parts or "venv" in parts or "node_modules" in parts:
            continue
        is_junk_dir = path.is_dir() and (
            path.name in DIR_NAMES or path.name.endswith(".egg-info")
        )
        is_junk_file = path.is_file() and (
            path.name in FILE_NAMES
            or path.name.startswith(".coverage.")
            or path.suffix in FILE_SUFFIXES
        )
        if is_junk_dir or is_junk_file:
            candidates.append(path)

    # 先删深路径，避免父目录已删导致子路径报错
    for path in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        if dry_run:
            print(f"would remove {path.relative_to(root)}")
            removed += 1
            continue
        if _rm_path(path):
            print(f"removed {path.relative_to(root)}")
            removed += 1

    print(f"cleaned {removed} path(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清除 pocket-agent 缓存与测试临时文件")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将删除的路径，不实际删除",
    )
    args = parser.parse_args(argv)
    return clean(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
