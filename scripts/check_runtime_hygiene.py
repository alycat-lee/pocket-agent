#!/usr/bin/env python3
"""运行时红线：src/ 不得出现 MockLLM / demo_tools / fixture_tools。"""

from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN = ("MockLLM", "demo_tools", "fixture_tools")


def iter_src_files(src_root: Path) -> list[Path]:
    files: list[Path] = []
    if not src_root.is_dir():
        return files
    for path in src_root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".yml", ".yaml"}:
            files.append(path)
    return files


def find_leaks(src_root: Path) -> list[str]:
    hits: list[str] = []
    for path in iter_src_files(src_root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(f"{path}: {needle}")
    return hits


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "src"
    leaks = find_leaks(root)
    if leaks:
        print("runtime hygiene failed:", file=sys.stderr)
        for item in leaks:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(f"runtime hygiene ok: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
