from __future__ import annotations

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import EMPTY_THINK_NOTE
from pocket_agent.runner import AgentRunner, RunResult, StepRecord, StopReason
from pocket_agent.session import SessionManager
from pocket_agent.task import TaskCard
from pocket_agent.tools import ToolRegistry, tool
from pocket_agent.verifier import (
    DEGRADE_PREFIX,
    attach_degrade,
    prompt_leak_phrases,
    verify,
)


def _card(
    type: str = "qa",
    *,
    total_chars: int | None = None,
    plan: list[str] | None = None,
) -> TaskCard:
    constraints = {}
    if total_chars is not None:
        constraints["total_chars"] = total_chars
    return TaskCard(
        type=type,  # type: ignore[arg-type]
        goal="测试",
        constraints=constraints,
        plan=plan or ["准备", "执行", "收尾"],
        needs_history=False,
    )


def test_prompt_leak_phrases_come_from_templates() -> None:
    phrases = prompt_leak_phrases()
    assert "可用工具" in phrases
    assert "当前进度" in phrases
    assert "只输出" in phrases


def test_think_leak_positive_and_negative() -> None:
    card = _card("qa")
    assert "思维链泄漏" in verify("<think>推理</think>最终答案", card)
    assert "思维链泄漏" in verify("嗯，我先看看这个问题怎么拆", card)
    assert "思维链泄漏" in verify("我现在需要调用工具", card)
    assert "思维链泄漏" in verify("首先，我得确认城市", card)
    assert verify("北京今天晴，气温二十五度左右。", card) == []


def test_prompt_leak_positive_and_negative() -> None:
    card = _card("qa")
    assert "prompt 泄漏" in verify("可用工具：get_weather\n当前进度：无", card)
    assert "prompt 泄漏" in verify("请只输出 JSON，不要解释。", card)
    assert verify("根据已有天气数据，北京晴。", card) == []
    # 长文叙事「只输出了…」不得误判为模板泄漏
    assert verify("他只输出了淡淡的笑，随即转身离开。" + ("正" * 20), _card("writer")) == []


def test_char_constraint_positive_and_negative() -> None:
    card = _card("writer", total_chars=200)
    assert "字数不足" in verify("短", card)
    long = "正" * 130
    assert "字数不足" not in verify(long, card)


def test_short_writer_and_qa_exception() -> None:
    assert "回答过短" in verify("太短了", _card("writer"))
    assert "回答过短" not in verify("好", _card("qa"))


def test_parrot_plan_positive_and_negative() -> None:
    plan = ["开篇建立人物", "中段推进冲突", "收束给出结局"]
    joined = "".join(plan)
    assert "复读计划" in verify(joined, _card("writer", plan=plan))
    assert "复读计划" not in verify(
        "林小满推开门，雨还在下，故事这才真正开始。",
        _card("writer", plan=plan),
    )


def test_agent_idle_tools_rejected_qa_writer_exempt() -> None:
    idle = "应调用工具但未调用"
    assert idle in verify(
        "北京大概晴吧。",
        _card("agent"),
        user_input="查北京天气",
        real_calls=0,
    )
    assert idle not in verify(
        "北京晴 25℃。",
        _card("agent"),
        user_input="查北京天气",
        real_calls=1,
    )
    assert idle not in verify(
        "北京大概晴吧。",
        _card("qa"),
        user_input="查北京天气",
        real_calls=0,
    )
    assert idle not in verify(
        "北京大概晴吧。",
        _card("writer"),
        user_input="查北京天气",
        real_calls=0,
    )


def test_now_clock_intent_is_agent(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=False, planner_enabled=False)
    llm = mock_llm_factory(["现在三点。", "报时"])

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            del event, payload

    mgr = SessionManager(
        cfg, llm, AgentRunner(cfg, llm, ToolRegistry(cfg)), events=_Bus()
    )
    mgr.chat("s1", "现在几点了")
    assert mgr.last_intent == "agent"


def test_weather_skip_tools_verify_rejects_and_retries(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, answer_max_retries=1)
    registry = ToolRegistry(cfg)

    @tool(registry, "查询指定城市天气")
    def get_weather(city: str) -> str:
        return f"{city}：晴，25℃"

    llm = mock_llm_factory(
        [
            {
                "type": "agent",
                "goal": "查北京天气",
                "constraints": {},
                "plan": ["查询", "回答"],
                "needs_history": False,
            },
            -1,
            "没查工具就猜北京晴。",
            "北京今天晴，气温二十五度左右，已经说明。",
            "天气查询",
        ]
    )
    rejected: list[list[str]] = []

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            if event == "answer.rejected":
                rejected.append(list(payload.get("reasons") or []))

    mgr = SessionManager(
        cfg, llm, AgentRunner(cfg, llm, registry), events=_Bus()
    )
    mgr.chat("s1", "查北京天气")
    assert rejected
    assert "应调用工具但未调用" in rejected[0]
    assert any("应调用工具但未调用" in p for p in llm.prompts)
    assert mgr.last_intent == "agent"


def test_execution_log_leak_rejected() -> None:
    card = _card("agent")
    leaked = '北京天气\n1. get_weather [缓存] args={"city":"北京"} → 晴'
    assert "执行日志泄漏" in verify(leaked, card)
    assert "执行日志泄漏" not in verify("北京今天晴，气温二十五度左右。", card)


def test_execution_log_real_d3_failures() -> None:
    card = _card("agent")
    plan_echo = (
        "1. 生成JSON文件：使用generate_json工具，参数设置为{...}，"
        "将结果保存到data/Beijing_weather.json中。\n\n"
        "2. 读取文件：调用read_file工具，"
    )
    placeholder = (
        "现在的时间是北京和上海的当地时间……"
        "（此处可能需要更多的信息，如您的位置或所需城市名称）"
    )
    reasons_plan = verify(plan_echo, card)
    assert "执行日志泄漏" in reasons_plan or "复读计划" in reasons_plan
    reasons_ph = verify(placeholder, card)
    assert "执行日志泄漏" in reasons_ph or "复读计划" in reasons_ph
    assert verify("北京：晴，25℃", card) == []


def test_quality_gate_real_dirty_answers() -> None:
    card = _card("agent")
    now_leak = (
        '现在无法报时。format_error: cannot parse int from '
        '\'{"value":\' leftover generate_int'
    )
    three_city = (
        "```json\n"
        "[\n"
        '  {"step": 1, "tool": "get_weather", "args": {"city": "北京"}},\n'
        '  {"step": 2, "tool": "get_weather", "args": {"city": "上海"}},\n'
        '  {"step": 3, "tool": "get_weather", "args": {"city": "广州"}},\n'
        '  {"step": 4'
    )
    shop = "2. **冷藏保存**：将新鲜的梨及时放入冰箱冷藏，以保持其新鲜和品质。"
    assert "执行日志泄漏" in verify(now_leak, card)
    reasons_plan = verify(three_city, card)
    assert "执行日志泄漏" in reasons_plan or "复读计划" in reasons_plan
    assert "格式违反" in verify(
        shop,
        _card("qa"),
        user_input="我一共买了几种水果？只输出整数。",
    )
    assert verify("北京：晴，25℃", card) == []
    assert verify("温差是7℃。", card) == []
    assert (
        verify("2", _card("qa"), user_input="只输出整数") == []
    )


def test_cache_marker_triggers_verify_retry(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, answer_max_retries=1)
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "查天气",
                "constraints": {},
                "plan": ["查", "答"],
                "needs_history": False,
            },
            "1. get_weather [缓存] args={\"city\":\"北京\"} → 晴",
            "北京今天晴，气温二十五度左右。",
            "天气",
        ]
    )
    rejected: list[list[str]] = []

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            if event == "answer.rejected":
                rejected.append(list(payload.get("reasons") or []))

    mgr = SessionManager(
        cfg, llm, AgentRunner(cfg, llm, ToolRegistry(cfg)), events=_Bus()
    )
    answer = mgr.chat("s1", "你好")
    assert rejected
    assert "执行日志泄漏" in rejected[0]
    assert "[缓存]" not in answer


def test_empty_think_note_retries_formal_answer(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, answer_max_retries=1)
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "报时",
                "constraints": {},
                "plan": ["答"],
                "needs_history": False,
            },
            EMPTY_THINK_NOTE,
            "现在是下午三点整，已经说明。",
            "报时",
        ]
    )
    rejected: list[list[str]] = []

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            if event == "answer.rejected":
                rejected.append(list(payload.get("reasons") or []))

    mgr = SessionManager(
        cfg, llm, AgentRunner(cfg, llm, ToolRegistry(cfg)), events=_Bus()
    )
    answer = mgr.chat("s1", "你好")
    assert rejected
    assert "正式回答为空" in rejected[0]
    assert EMPTY_THINK_NOTE not in answer
    assert "下午三点" in answer


def test_degrade_message() -> None:
    text = attach_degrade("正文", ["思维链泄漏", "字数不足"])
    assert text.startswith("正文")
    assert DEGRADE_PREFIX in text
    assert "思维链泄漏" in text
    assert "字数不足" in text


def test_verifier_retry_then_degrade(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, answer_max_retries=1)
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "闲聊问候",
                "constraints": {},
                "plan": ["问候", "回应", "结束"],
                "needs_history": False,
            },
            "<think>先想想</think>不该给用户",
            "你好，今天过得怎么样，有什么想聊的？",
            "标题问候",
        ]
    )
    runner = AgentRunner(cfg, llm, ToolRegistry(cfg))
    rejected: list[list[str]] = []

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            if event == "answer.rejected":
                rejected.append(list(payload.get("reasons") or []))

    mgr = SessionManager(cfg, llm, runner, events=_Bus())
    answer = mgr.chat("s1", "你好")
    assert rejected
    assert "思维链泄漏" in rejected[0]
    assert "<think>" not in answer
    assert "今天过得怎么样" in answer


def test_verifier_retry_exhausted_appends_degrade(mock_llm_factory: type) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True, answer_max_retries=1)
    llm = mock_llm_factory(
        [
            {
                "type": "qa",
                "goal": "闲聊",
                "constraints": {},
                "plan": ["a", "b", "c"],
                "needs_history": False,
            },
            "<think>一轮</think>坏答案",
            "<think>二轮</think>还是坏",
            "标题",
        ]
    )
    runner = AgentRunner(cfg, llm, ToolRegistry(cfg))

    class _Bus:
        def emit(self, event: str, payload: dict) -> None:
            del event, payload

    mgr = SessionManager(cfg, llm, runner, events=_Bus())
    answer = mgr.chat("s1", "你好")
    assert DEGRADE_PREFIX in answer
    assert "思维链泄漏" in answer
    assert mgr.last_verify_reasons == ["思维链泄漏"]


def test_format_violation_fallback_extracts_first_digit(
    mock_llm_factory: type,
) -> None:
    cfg = AgentConfig(max_steps=3, verifier_enabled=True)
    llm = mock_llm_factory(["占位"])
    mgr = SessionManager(cfg, llm, AgentRunner(cfg, llm, ToolRegistry(cfg)))
    mgr.last_result = RunResult(
        answer="脏",
        steps=[
            StepRecord(
                step=1,
                tool_name="shop",
                args={},
                result="苹果、梨，共2种",
                from_cache=False,
            )
        ],
        stop_reason=StopReason.MODEL_DONE,
        stats={},
    )
    cleaned = mgr._fallback_answer(
        "2. **冷藏保存**：将新鲜的梨及时放入冰箱冷藏。",
        ["格式违反"],
    )
    assert cleaned == "2"
