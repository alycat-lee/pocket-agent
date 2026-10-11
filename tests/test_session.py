from __future__ import annotations

from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.runner import AgentRunner
from pocket_agent.session import SessionManager
from pocket_agent.tools import ToolRegistry, tool


def _manager(
    mock_llm_factory: type,
    responses: list[object],
    config: AgentConfig | None = None,
    *,
    with_echo_tool: bool = False,
):
    """with_echo_tool=True 时注册一个占位工具，强制走 choose→final 双调用路径。"""
    cfg = config or AgentConfig(max_steps=5)
    registry = ToolRegistry(cfg)
    if with_echo_tool:

        @tool(registry, "回声工具")
        def echo(text: str) -> str:
            return text

    llm = mock_llm_factory(responses)
    runner = AgentRunner(cfg, llm, registry)
    return SessionManager(cfg, llm, runner), llm, registry


def test_rewrite_called(mock_llm_factory: type) -> None:
    # 空工具表：每轮仅 final generate；第 2 轮前多一次 rewrite
    mgr, llm, _ = _manager(
        mock_llm_factory,
        ["第一轮回答", "查询上海的天气", "第二轮回答"],
    )
    mgr.chat("s1", "北京天气")
    assert mgr.rewrite_call_count == 0
    assert not any("把用户的最新输入改写" in p for p in llm.prompts)

    mgr.chat("s1", "那上海呢")
    assert mgr.rewrite_call_count == 1
    assert any("把用户的最新输入改写" in p for p in llm.prompts)


def test_pronoun_follow(mock_llm_factory: type) -> None:
    rewritten = "查询上海的天气"
    mgr, _, _ = _manager(
        mock_llm_factory,
        ["北京晴 25℃", rewritten, "上海多云 18℃"],
    )
    mgr.chat("s1", "北京天气")
    mgr.chat("s1", "那上海呢")
    assert mgr.last_rewritten == rewritten
    assert rewritten in mgr.last_run_task


def test_rolling_summary_trigger(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=3, keep_full_turns=6, session_summary_limit=500)
    # 空工具：turn1(final)；turn2..6(rewrite,final)；
    # turn7(rewrite,final,summary)；turn8(rewrite,final,summary)
    responses: list[object] = ["回答1"]
    for i in range(2, 9):
        responses.append(f"完整问题{i}")
        responses.append(f"回答{i}")
        if i >= 7:
            responses.append(f"摘要覆盖到第{i}轮关键事实")

    mgr, llm, _ = _manager(mock_llm_factory, responses, config)
    for i in range(1, 9):
        mgr.chat("s1", f"问题{i}")
        if i == 6:
            assert mgr.summary_call_count == 0
            assert mgr.get_state("s1").rolling_summary == ""
        if i == 7:
            assert mgr.summary_call_count >= 1
            summary = mgr.get_state("s1").rolling_summary
            assert summary
            assert len(summary) <= config.session_summary_limit
            assert any("压缩为不超过" in p for p in llm.prompts)

    assert len(mgr.get_state("s1").turns) == config.keep_full_turns


def test_session_isolation(mock_llm_factory: type) -> None:
    mgr, llm, _ = _manager(
        mock_llm_factory,
        [
            "A的秘密是苹果",
            "B的秘密是香蕉",
            "用户A在问自己喜欢什么",
            "苹果",
            "用户B在问自己喜欢什么",
            "香蕉",
        ],
    )
    mgr.chat("alice", "我喜欢苹果")
    mgr.chat("bob", "我喜欢香蕉")
    mgr.chat("alice", "我喜欢什么")
    alice_rewrite_prompts = [p for p in llm.prompts if "最新输入：我喜欢什么" in p]
    assert alice_rewrite_prompts
    assert "香蕉" not in alice_rewrite_prompts[0]
    assert "苹果" in alice_rewrite_prompts[0]

    mgr.chat("bob", "我喜欢什么")
    bob_prompts = [p for p in llm.prompts if "最新输入：我喜欢什么" in p]
    assert len(bob_prompts) >= 2
    assert "苹果" not in bob_prompts[-1]
    assert "香蕉" in bob_prompts[-1]


def test_write_op_cross_turn(mock_llm_factory: type) -> None:
    config = AgentConfig(max_steps=8)
    registry = ToolRegistry(config)
    write_calls = {"n": 0}
    read_calls = {"n": 0}

    @tool(registry, "写入笔记", cacheable=False)
    def write_note(text: str) -> str:
        write_calls["n"] += 1
        return f"wrote:{text}"

    @tool(registry, "读取笔记", cacheable=True)
    def read_note(key: str) -> str:
        read_calls["n"] += 1
        return f"read:{key}"

    responses: list[object] = [
        0,
        {"text": "secret"},
        1,
        {"key": "k1"},
        -1,
        "第一轮完成",
        "请再次写入并读取笔记",
        0,
        {"text": "secret"},
        1,
        {"key": "k1"},
        -1,
        "第二轮完成",
    ]
    llm = mock_llm_factory(responses)
    runner = AgentRunner(config, llm, registry)
    mgr = SessionManager(config, llm, runner)

    mgr.chat("s1", "写入并读取")
    assert write_calls["n"] == 1
    assert read_calls["n"] == 1

    mgr.chat("s1", "再来一次同样操作")
    assert write_calls["n"] == 1
    assert read_calls["n"] == 2

    assert mgr.last_result is not None
    write_steps = [s for s in mgr.last_result.steps if s.tool_name == "write_note"]
    read_steps = [s for s in mgr.last_result.steps if s.tool_name == "read_note"]
    assert write_steps and write_steps[0].from_cache is True
    assert read_steps and read_steps[0].from_cache is False


def test_auto_rename_after_first_turn(mock_llm_factory: type) -> None:
    """首轮后自动生成标题；emit session.renamed 事件。"""
    cfg = AgentConfig(max_steps=3)
    registry = ToolRegistry(cfg)
    llm = mock_llm_factory([1, "回答1", "北京天气查询"])  # 1=intent→agent
    runner = AgentRunner(cfg, llm, registry)

    renamed_events: list[dict] = []

    class _CaptureEvents:
        def emit(self, event: str, payload: dict) -> None:
            if event == "session.renamed":
                renamed_events.append(payload)

    mgr = SessionManager(cfg, llm, runner, events=_CaptureEvents())
    mgr.chat("s1", "北京天气怎么样")

    state = mgr.get_state("s1")
    assert state.title == "北京天气查询"
    assert len(renamed_events) == 1
    assert renamed_events[0]["session_id"] == "s1"
    assert renamed_events[0]["title"] == "北京天气查询"
    # list_sessions 返回 title
    rows = mgr.list_sessions()
    assert rows[0]["title"] == "北京天气查询"


def test_auto_rename_fallback_on_llm_failure(mock_llm_factory: type) -> None:
    """标题生成失败时回退为用户输入前 12 字截断。"""
    cfg = AgentConfig(max_steps=3)
    registry = ToolRegistry(cfg)
    # 1=intent→agent, "回答1"=final, RuntimeError=title gen fail
    llm = mock_llm_factory([1, "回答1", RuntimeError("title gen failed")])
    runner = AgentRunner(cfg, llm, registry)

    class _NullEvents:
        def emit(self, event: str, payload: dict) -> None: ...

    mgr = SessionManager(cfg, llm, runner, events=_NullEvents())
    mgr.chat("s1", "这是一个非常非常长的用户输入超过十二个字呢")

    state = mgr.get_state("s1")
    assert state.title == "这是一个非常非常长的用户输入超过十二个字呢"[:12]


def test_no_auto_rename_without_events(mock_llm_factory: type) -> None:
    """无事件总线时不触发标题生成（不消耗 MockLLM 队列）。"""
    cfg = AgentConfig(max_steps=3)
    registry = ToolRegistry(cfg)
    llm = mock_llm_factory(["回答1"])
    runner = AgentRunner(cfg, llm, registry)
    mgr = SessionManager(cfg, llm, runner)  # events=None
    mgr.chat("s1", "北京天气")
    assert mgr.get_state("s1").title == "新会话"


def test_intent_detected_is_first_and_payload_complete(
    mock_llm_factory: type,
) -> None:
    """intent.detected 是每轮第一条，且含 source/reason/turn。"""
    cfg = AgentConfig(max_steps=3)
    registry = ToolRegistry(cfg)
    llm = mock_llm_factory(["晴", "问候"])
    runner = AgentRunner(cfg, llm, registry)
    captured: list[tuple[str, dict]] = []

    class _Cap:
        def emit(self, event: str, payload: dict) -> None:
            captured.append((event, payload))

    mgr = SessionManager(cfg, llm, runner, events=_Cap())
    mgr.chat("s1", "你好")

    assert captured
    assert captured[0][0] == "task.planned"
    assert captured[0][1]["type"] == "qa"
    intent = next(p for n, p in captured if n == "intent.detected")
    for key in ("session_id", "mode", "confidence", "source", "reason", "turn"):
        assert key in intent
    assert intent["session_id"] == "s1"
    assert intent["mode"] == "qa"
    assert intent["source"] == "rule"
    assert intent["reason"] == "短句闲聊"
    assert intent["turn"] == 1


def test_force_mode_manual_reuses_rewritten(mock_llm_factory: type) -> None:
    """force_mode 纠正：source=manual，复用 query_rewritten，不重走 rewrite。"""
    cfg = AgentConfig(max_steps=3)
    registry = ToolRegistry(cfg)
    llm = mock_llm_factory([1, "第一轮", "标题", "查询上海天气", "第二轮", "纠正后"])
    runner = AgentRunner(cfg, llm, registry)
    captured: list[tuple[str, dict]] = []

    class _Cap:
        def emit(self, event: str, payload: dict) -> None:
            captured.append((event, payload))

    mgr = SessionManager(cfg, llm, runner, events=_Cap())
    mgr.chat("s1", "北京天气")
    mgr.chat("s1", "那上海呢")
    rewrite_after_second = mgr.rewrite_call_count
    rewritten = mgr.last_rewritten

    mgr.chat("s1", "那上海呢", force_mode="agent")
    assert mgr.rewrite_call_count == rewrite_after_second
    assert mgr.last_rewritten == rewritten
    assert mgr.last_intent == "agent"
    intents = [p for n, p in captured if n == "intent.detected"]
    assert intents[-1]["source"] == "manual"
    assert intents[-1]["reason"] == "用户纠正"
    assert intents[-1]["mode"] == "agent"
    assert len(mgr.get_state("s1").turns) == 2


def test_compose_excludes_unrelated_history_for_writer_and_qa(
    mock_llm_factory: type,
) -> None:
    """跨轮无关历史不得进入 writer/qa 的任务 prompt。"""
    cfg = AgentConfig(max_steps=3, verifier_enabled=False)
    registry = ToolRegistry(cfg)
    llm = mock_llm_factory([1, "上海多云 18℃", "天气标题", 0, 0, "闲聊回答"])
    runner = AgentRunner(cfg, llm, registry)

    class _Writer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def write(self, topic: str, **kwargs: Any) -> str:
            self.calls.append((topic, dict(kwargs)))
            return "小说正文若干字"

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            del event, payload

    writer = _Writer()
    mgr = SessionManager(cfg, llm, runner, events=_Bus(), writer=writer)
    mgr.chat("s1", "查询上海天气")
    mgr.chat("s1", "给我生成一篇2万字的小说 内容随意")

    assert mgr.last_intent == "writer"
    assert "上海" not in mgr.last_run_task
    assert writer.calls
    topic, kwargs = writer.calls[0]
    assert "上海" not in topic
    assert kwargs.get("total_chars") == 20000
    for prompt in llm.prompts:
        if "你正在逐段撰写文章" in prompt or "为主题「" in prompt:
            assert "上海" not in prompt

    mgr.chat("s1", "你好")
    assert mgr.last_intent == "qa"
    assert "上海" not in mgr.last_run_task
    for prompt in llm.prompts:
        if "请回答：你好" in prompt or prompt.strip() == "你好":
            assert "上海" not in prompt
        if "基于以上信息回答任务" in prompt and "你好" in prompt:
            assert "上海" not in prompt
