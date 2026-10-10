from __future__ import annotations

import argparse
import json
import logging
import sys

from pocket_agent.config import AgentConfig
from pocket_agent.demo_tools import register_demo_tools
from pocket_agent.llm_client import LlamaCppClient, LLMConnectionError
from pocket_agent.runner import AgentRunner, RunResult
from pocket_agent.tools import ToolRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pocket-agent", description="pocket-agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="运行内置演示工具任务")
    demo_parser.add_argument("task", help="任务文本")
    demo_parser.add_argument(
        "--base-url",
        default=AgentConfig().base_url,
        help="llama.cpp server OpenAI 兼容地址",
    )
    demo_parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 DEBUG 级内部日志",
    )
    demo_parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="启用断点续跑：指定 run_id，中断后可用同一 id 恢复",
    )

    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo(
            task=args.task,
            base_url=args.base_url,
            verbose=args.verbose,
            run_id=args.resume,
        )
    return 1


def run_demo(
    *,
    task: str,
    base_url: str,
    verbose: bool,
    run_id: str | None = None,
) -> int:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    config = AgentConfig(base_url=base_url)
    registry = ToolRegistry(config)
    register_demo_tools(registry)
    llm = LlamaCppClient(config)
    runner = AgentRunner(config, llm, registry)

    try:
        result = runner.run(task, run_id=run_id)
    except LLMConnectionError as exc:
        print(f"连接 LLM 失败：{exc}", file=sys.stderr)
        print("请确认已启动 llama-server（见 README）。", file=sys.stderr)
        return 2

    _print_steps(result)
    print()
    print("=== 最终回答 ===")
    print(result.answer)
    print()
    print("=== 汇总 ===")
    print(f"stop_reason={result.stop_reason.value}")
    for key in (
        "total_steps",
        "real_calls",
        "cache_hits",
        "intercepts",
        "prompt_tokens_max",
    ):
        print(f"{key}={result.stats.get(key)}")
    return 0


def _print_steps(result: RunResult) -> None:
    tokens = result.stats.get("prompt_tokens_max", 0)
    for step in result.steps:
        cache = "hit" if step.from_cache else "miss"
        args_text = json.dumps(step.args, ensure_ascii=False, sort_keys=True)
        print(
            f"[Step {step.step}] choose={step.tool_name} "
            f"args={args_text} cache={cache} tokens≈{tokens}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
