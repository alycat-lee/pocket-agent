from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMConnectionError
from pocket_agent.plugins import builtin_plugins
from pocket_agent.runner import RunResult
from pocket_agent.runtime import assemble, load_pocket_config

_DEFAULT_POCKET = Path(__file__).resolve().parent / "pocket.yml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pocket-agent", description="pocket-agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="连接真实模型跑真实任务")
    demo_parser.add_argument("task", help="任务文本")
    demo_parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖 pocket.yml 中 llm-llamacpp 的 base_url",
    )
    demo_parser.add_argument(
        "--config",
        default=str(_DEFAULT_POCKET),
        help="pocket.yml 路径（默认仓库根目录）",
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

    serve_parser = subparsers.add_parser(
        "serve",
        help="按 pocket.yml 装配并保持运行（web profile 下提供 UI）",
    )
    serve_parser.add_argument(
        "--config",
        default=str(_DEFAULT_POCKET),
        help="pocket.yml 路径（默认仓库根目录）",
    )
    serve_parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖 llm-llamacpp 的 base_url",
    )
    serve_parser.add_argument(
        "--profile",
        choices=("headless", "web", "tui"),
        default=None,
        help="覆盖 pocket.yml 的 profile（web 加载浏览器 UI，tui 进入终端界面）",
    )
    serve_parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 DEBUG 级内部日志",
    )

    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo(
            task=args.task,
            base_url=args.base_url,
            verbose=args.verbose,
            run_id=args.resume,
            config_path=args.config,
        )
    if args.command == "serve":
        return run_serve(
            config_path=args.config,
            base_url=args.base_url,
            profile=args.profile,
            verbose=args.verbose,
        )
    return 1


def run_demo(
    *,
    task: str,
    base_url: str | None,
    verbose: bool,
    run_id: str | None = None,
    config_path: str | Path = _DEFAULT_POCKET,
) -> int:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    data = _prepare_config(config_path, base_url=base_url, verbose=verbose)
    ctx = assemble(builtin_plugins(), data)
    try:
        agent = ctx.require("agent")
        try:
            result = agent.run(task, run_id=run_id)
        except LLMConnectionError as exc:
            print(f"模型未连接：{exc}", file=sys.stderr)
            print(
                "请先启动 llama-server，或在设置页添加可用的 OpenAI 兼容端点后再试。",
                file=sys.stderr,
            )
            return 2
    finally:
        ctx.unmount_all()

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


def run_serve(
    *,
    config_path: str | Path = _DEFAULT_POCKET,
    base_url: str | None = None,
    profile: str | None = None,
    verbose: bool = False,
) -> int:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    data = _prepare_config(
        config_path,
        base_url=base_url,
        verbose=verbose,
        profile=profile,
    )
    ctx = assemble(builtin_plugins(), data)
    try:
        if ctx.has("tui"):
            ctx.require("tui").run()
            return 0
        if ctx.has("http"):
            http = ctx.require("http")
            print(
                f"Web UI: http://{http['host']}:{http['port']}/",
                flush=True,
            )
        else:
            print(
                "已装配（无 ui-web）。将 pocket.yml 的 profile 设为 web，"
                "或使用：python cli.py serve --profile web",
                flush=True,
            )
        print("Ctrl+C 停止并卸载全部插件。", flush=True)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n正在卸载…", flush=True)
    finally:
        ctx.unmount_all()
    return 0


def _prepare_config(
    config_path: str | Path,
    *,
    base_url: str | None,
    verbose: bool,
    profile: str | None = None,
) -> dict[str, Any]:
    data = load_pocket_config(config_path)
    if profile is not None:
        data["profile"] = profile
    plugins = data.get("plugins")
    if not isinstance(plugins, list):
        raise SystemExit("pocket.yml plugins must be a list")

    resolved_base = base_url or AgentConfig().base_url
    plugin_configs: dict[str, dict[str, Any]] = {}
    for entry in plugins:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        cfg = entry.setdefault("config", {})
        if not isinstance(cfg, dict):
            entry["config"] = {}
            cfg = entry["config"]
        if name == "llm-llamacpp" and base_url is not None:
            cfg["base_url"] = resolved_base
        if name == "ui-cli" and verbose:
            cfg["verbose"] = True
        if isinstance(name, str):
            plugin_configs[name] = dict(cfg)
    for entry in plugins:
        if not isinstance(entry, dict) or entry.get("name") != "ui-web":
            continue
        cfg = entry.setdefault("config", {})
        if not isinstance(cfg, dict):
            entry["config"] = {}
            cfg = entry["config"]
        cfg["pocket_data"] = data
        cfg["pocket_path"] = str(config_path)
        cfg["plugin_catalog"] = builtin_plugins()
        cfg["plugin_configs"] = plugin_configs
    return data


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
