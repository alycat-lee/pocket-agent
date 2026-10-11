from __future__ import annotations

from pathlib import Path
from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.events import EventStore
from pocket_agent.plugins.ui_web import attach_event_persistence
from pocket_agent.providers import (
    ProviderConfig,
    ProviderRegistry,
    ProvidersConfig,
    RoutingConfig,
)
from pocket_agent.runner import AgentRunner
from pocket_agent.runtime import Context
from pocket_agent.session import SessionManager
from pocket_agent.tools import ToolRegistry
from pocket_agent.writer import LongWriter


class _StreamingFake:
    """测试用流式 LLM：generate_stream 按块回调，不进入 src/。"""

    def __init__(
        self,
        answers: list[str],
        *,
        ints: list[int] | None = None,
        jsons: list[dict[str, Any]] | None = None,
        chunk: int = 8,
    ) -> None:
        self._answers = list(answers)
        self._ints = list(ints or [])
        self._jsons = list(jsons or [])
        self.prompts: list[str] = []
        self.chunk = chunk

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._answers:
            raise AssertionError("StreamingFake answer queue exhausted")
        return self._answers.pop(0)

    def generate_stream(self, prompt: str, on_delta: Any) -> str:
        text = self.generate(prompt)
        size = max(1, self.chunk)
        if not text:
            return text
        for i in range(0, len(text), size):
            on_delta(text[i : i + size])
        return text

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        self.prompts.append(prompt)
        if self._ints:
            return int(self._ints.pop(0))
        return -1

    def generate_json(self, prompt: str, schema: dict) -> dict:
        self.prompts.append(prompt)
        if self._jsons:
            return dict(self._jsons.pop(0))
        return {}


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))


class _CtxBus:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))
        self._ctx.emit(event, payload)


def _long(prefix: str) -> str:
    # 64~80 字：超过 32 字泵阈值两次，又避开 writer 续写/裁剪。
    pad = "增量文字用来撑开缓冲。"
    body = prefix + pad * 4
    return body[:72]


def _mgr(
    llm: _StreamingFake,
    bus: _Bus | _CtxBus,
    *,
    writer: LongWriter | None = None,
    cfg: AgentConfig | None = None,
) -> SessionManager:
    config = cfg or AgentConfig(
        max_steps=2,
        planner_enabled=False,
        verifier_enabled=False,
    )
    runner = AgentRunner(config, llm, ToolRegistry(config), events=bus)
    return SessionManager(config, llm, runner, events=bus, writer=writer)


def _deltas(bus: _Bus | _CtxBus) -> list[str]:
    return [str(p.get("delta") or "") for n, p in bus.events if n == "answer.delta"]


def _done_answer(bus: _Bus | _CtxBus) -> str:
    for name, payload in reversed(bus.events):
        if name == "answer.done":
            return str(payload.get("answer") or "")
    raise AssertionError("missing answer.done")


def test_provider_registry_stream_falls_back_to_generate() -> None:
    class _Plain:
        def generate(self, prompt: str) -> str:
            return "整段回答"

        def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
            return -1

        def generate_json(self, prompt: str, schema: dict) -> dict:
            return {}

    cfg = ProvidersConfig(
        providers=[
            ProviderConfig(
                id="p1",
                name="local",
                type="llamacpp",
                base_url="http://127.0.0.1:9/v1",
            )
        ],
        default_provider_id="p1",
        routing=RoutingConfig(choose_fill="p1", generate="p1"),
    )
    registry = ProviderRegistry(cfg, AgentConfig())
    registry._clients["p1"] = _Plain()  # type: ignore[assignment]
    got: list[str] = []
    assert registry.generate_stream("问", got.append) == "整段回答"
    assert got == ["整段回答"]


def test_answer_delta_is_not_persisted(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events")
    ctx = Context()
    attach_event_persistence(ctx, store)
    bus = _CtxBus(ctx)
    llm = _StreamingFake([_long("我是本地助手口袋代理"), "身份"])
    mgr = _mgr(llm, bus)
    mgr.chat("s1", "你是谁", force_mode="qa")
    path = tmp_path / "events" / "s1.jsonl"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert "answer.delta" not in raw
    assert "answer.done" in raw
    assert any(name == "answer.delta" for name, _ in bus.events)


def test_qa_path_streams_deltas() -> None:
    bus = _Bus()
    text = _long("我是口袋代理默认助手")
    llm = _StreamingFake([text, "标题"])
    mgr = _mgr(llm, bus)
    answer = mgr.chat("s1", "你是谁", force_mode="qa")
    deltas = _deltas(bus)
    assert len(deltas) >= 2
    assert "".join(deltas) == text
    assert _done_answer(bus) == text
    assert answer == text


def test_verify_retry_streams_deltas() -> None:
    bus = _Bus()
    retry = "口袋代理是运行在用户本地的助手，回答应当完整清楚。" * 3
    llm = _StreamingFake(["短", retry, "标题"])
    cfg = AgentConfig(
        max_steps=2,
        planner_enabled=False,
        verifier_enabled=True,
        answer_max_retries=1,
    )
    mgr = _mgr(llm, bus, cfg=cfg)
    answer = mgr.chat("s1", "请用100字说明口袋代理", force_mode="qa")
    rejected_at = next(
        i for i, (n, _) in enumerate(bus.events) if n == "answer.rejected"
    )
    retry_deltas = [
        str(p.get("delta") or "")
        for n, p in bus.events[rejected_at + 1 :]
        if n == "answer.delta"
    ]
    assert len(retry_deltas) >= 2
    assert "".join(retry_deltas) == retry
    assert _done_answer(bus) == retry
    assert answer == retry


def test_runner_path_streams_deltas() -> None:
    bus = _Bus()
    text = _long("工具路径最终回答也要拆成多条增量")
    llm = _StreamingFake([text, "标题"], ints=[-1])
    mgr = _mgr(llm, bus)
    answer = mgr.chat("s1", "随便聊一句", force_mode="agent")
    deltas = _deltas(bus)
    assert len(deltas) >= 2
    assert "".join(deltas) == text
    assert _done_answer(bus) == text
    assert answer == text


def test_writer_path_streams_section_breaks() -> None:
    bus = _Bus()
    first = ("甲段专属开篇" + "甲" * 60)[:72]
    second = ("乙段专属收束" + "乙" * 60)[:72]
    llm = _StreamingFake(
        [first, second, "补写一", "补写二", "标题"],
        jsons=[
            {
                "type": "writer",
                "goal": "短文",
                "constraints": {},
                "plan": ["开篇", "收束"],
                "needs_history": False,
                "needs_knowledge": False,
            }
        ],
    )
    cfg = AgentConfig(
        max_steps=2,
        planner_enabled=True,
        verifier_enabled=False,
        section_char_target=80,
    )
    writer = LongWriter(cfg, llm)
    mgr = _mgr(llm, bus, writer=writer, cfg=cfg)
    answer = mgr.chat("s1", "写一篇160字的短文")
    deltas = _deltas(bus)
    joined = "".join(deltas)
    assert len(deltas) >= 2
    assert "\n\n" in joined
    assert _done_answer(bus) == answer
    assert first in joined
    assert second in joined
