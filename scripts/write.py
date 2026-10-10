#!/usr/bin/env python3
"""长文本生成 CLI：大纲 → 逐段撰写 → 拼接去重，并输出连贯性指标。"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LlamaCppClient, LLMConnectionError
from pocket_agent.writer import LongWriter, SectionSpec, WriterMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pocket-agent 长文本生成")
    parser.add_argument("topic", help="文章主题")
    parser.add_argument(
        "--chars",
        type=int,
        default=3000,
        help="目标总字数（默认 3000）",
    )
    parser.add_argument(
        "--material",
        type=Path,
        default=None,
        help="参考材料文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 Markdown 路径（默认只打印）",
    )
    parser.add_argument(
        "--base-url",
        default=AgentConfig().base_url,
        help="llama.cpp server OpenAI 兼容地址（默认 http://localhost:8000/v1）",
    )
    args = parser.parse_args(argv)

    material = ""
    if args.material is not None:
        path = args.material
        if not path.is_file():
            path = REPO_ROOT / args.material
        if not path.is_file():
            print(f"材料文件不存在：{args.material}", file=sys.stderr)
            return 2
        material = path.read_text(encoding="utf-8")

    # 思考模型会先占 completion 预算写 reasoning，留足余量避免 content 被截空
    config = AgentConfig(
        base_url=args.base_url,
        max_tokens=max(1024, args.chars // 2),
        timeout=180.0,
    )
    llm = LlamaCppClient(config)
    writer = LongWriter(config, llm)

    def _on_section(index: int, section: SectionSpec, body: str) -> None:
        print(f"\n===== 第 {index} 段 · {section.title} =====\n")
        print(body)
        sys.stdout.flush()

    writer.on_section = _on_section

    print(f"主题：{args.topic}")
    print(f"目标字数：{args.chars}")
    try:
        article = writer.write(args.topic, total_chars=args.chars, material=material)
    except LLMConnectionError as exc:
        print(f"连接 LLM 失败：{exc}", file=sys.stderr)
        return 2

    metrics = WriterMetrics()
    sections = writer.last_sections
    pair_overlaps = [
        metrics.ngram_overlap(a, b, n=3) for a, b in itertools.pairwise(sections)
    ]
    max_pair = max(pair_overlaps) if pair_overlaps else 0.0
    conflicts = metrics.numeric_consistency(article)
    coverage = metrics.outline_coverage(writer.last_outline, article)

    print("\n===== 连贯性指标 =====")
    print(f"段间 3-gram 重叠率（最大）: {max_pair:.2%}")
    print(f"数字一致性冲突数: {len(conflicts)}")
    if conflicts:
        for item in conflicts:
            print(f"  - {item}")
    print(f"大纲覆盖率: {coverage:.2%}")
    print(f"总字数: {len(article)}")

    if args.output is not None:
        out = args.output if args.output.is_absolute() else Path.cwd() / args.output
        out.write_text(article, encoding="utf-8")
        print(f"\n已写入: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
