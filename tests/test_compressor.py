from __future__ import annotations

import math
import re

import pytest

from pocket_agent.compressor import ContextCompressor
from pocket_agent.config import AgentConfig


class StubLLM:
    """仅用于 compress_result 的 LLM 摘要路径测试。"""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        raise AssertionError("generate_int should not be called")

    def generate_json(self, prompt: str, schema: dict) -> dict:
        raise AssertionError("generate_json should not be called")


@pytest.fixture
def compressor() -> ContextCompressor:
    return ContextCompressor(AgentConfig())


def test_estimate_tokens_ceil_division(compressor: ContextCompressor) -> None:
    assert compressor.estimate_tokens("") == 0
    assert compressor.estimate_tokens("a") == 1
    assert compressor.estimate_tokens("ab") == 2  # ceil(2/1.5)=2
    assert compressor.estimate_tokens("abc") == 2  # ceil(3/1.5)=2
    text = "x" * 100
    assert compressor.estimate_tokens(text) == math.ceil(100 / 1.5)


def test_estimate_tokens_monotonic(compressor: ContextCompressor) -> None:
    shorter = "你好世界"
    longer = shorter + "额外内容"
    assert compressor.estimate_tokens(longer) >= compressor.estimate_tokens(shorter)


def test_compress_result_under_limit_unchanged(compressor: ContextCompressor) -> None:
    text = "短结果"
    assert compressor.compress_result(text) == text


def test_compress_result_rule_truncate_length_and_format() -> None:
    config = AgentConfig(per_result_limit=300)
    compressor = ContextCompressor(config)
    limit = config.per_result_limit
    text = "A" * 200 + "M" * 400 + "Z" * 200
    out = compressor.compress_result(text)
    assert len(out) <= limit
    match = re.search(r"\n\[\.\.\.截断 (\d+) 字\.\.\.\]\n", out)
    assert match is not None
    omitted = int(match.group(1))
    assert omitted > 0
    head, _, tail = out.partition(match.group(0))
    assert head.startswith("A")
    assert tail.endswith("Z")
    assert len(head) + omitted + len(tail) == len(text)


def test_compress_result_llm_none_forces_rule_truncate() -> None:
    config = AgentConfig(use_llm_summarize=True, per_result_limit=80)
    comp = ContextCompressor(config, llm=None)
    text = "前" * 100 + "中" * 200 + "后" * 100
    out = comp.compress_result(text)
    assert len(out) <= 80
    assert "截断" in out


def test_compress_result_llm_summarize_path() -> None:
    config = AgentConfig(use_llm_summarize=True, per_result_limit=50)
    llm = StubLLM("摘要：温度25")
    comp = ContextCompressor(config, llm=llm)
    text = "原始超长内容" * 100
    out = comp.compress_result(text)
    assert out == "摘要：温度25"
    assert len(llm.prompts) == 1
    assert "原始超长内容" in llm.prompts[0] or "截断" in llm.prompts[0]


def test_compress_result_llm_summary_still_truncated() -> None:
    config = AgentConfig(use_llm_summarize=True, per_result_limit=40)
    llm = StubLLM("S" * 200)
    comp = ContextCompressor(config, llm=llm)
    out = comp.compress_result("X" * 500)
    assert len(out) <= 40
    assert "截断" in out


def test_build_prompt_keeps_fixed_sections() -> None:
    config = AgentConfig(prompt_char_budget=200)
    comp = ContextCompressor(config)
    system = "SYS" * 20
    task = "TASK" * 20
    state = "STATE" * 20
    step_info = "STEP" * 20
    prompt = comp.build_prompt(
        system=system,
        task=task,
        state=state,
        step_info=step_info,
        history=["old", "new"],
    )
    # 固定区段截断后仍出现在结果中
    assert prompt.startswith(system[:300])
    assert "TASK" in prompt
    assert "STATE" in prompt
    assert "STEP" in prompt


def test_build_prompt_history_newest_first_fill_chronological_order() -> None:
    config = AgentConfig(prompt_char_budget=5000)
    comp = ContextCompressor(config)
    history = ["step-1-oldest", "step-2", "step-3-newest"]
    prompt = comp.build_prompt(
        system="s",
        task="t",
        state="st",
        step_info="si",
        history=history,
    )
    pos1 = prompt.index("step-1-oldest")
    pos2 = prompt.index("step-2")
    pos3 = prompt.index("step-3-newest")
    assert pos1 < pos2 < pos3


def test_build_prompt_drops_oldest_when_over_budget() -> None:
    # 固定区段很短，预算只够最新两条历史
    system, task, state, step_info = "S", "T", "A", "I"
    fixed = f"{system}\n\n{task}\n\n{state}\n\n{step_info}"
    h1 = "H1" + ("o" * 40)  # oldest
    h2 = "H2" + ("m" * 40)
    h3 = "H3" + ("n" * 40)  # newest
    # fixed + sep + h2 + sep + h3 刚好放入；再加 h1 超预算
    sep = "\n\n"
    budget = len(fixed) + len(sep) + len(h3) + len(sep) + len(h2)
    config = AgentConfig(prompt_char_budget=budget)
    comp = ContextCompressor(config)
    prompt = comp.build_prompt(
        system=system,
        task=task,
        state=state,
        step_info=step_info,
        history=[h1, h2, h3],
    )
    assert h1 not in prompt
    assert h2 in prompt
    assert h3 in prompt
    assert prompt.index(h2) < prompt.index(h3)


def test_build_prompt_fixed_always_kept_even_if_over_budget() -> None:
    config = AgentConfig(prompt_char_budget=10)
    comp = ContextCompressor(config)
    prompt = comp.build_prompt(
        system="system-section",
        task="task-section",
        state="state-section",
        step_info="step-section",
        history=["hist-should-drop"],
    )
    assert "system-section" in prompt
    assert "task-section" in prompt
    assert "state-section" in prompt
    assert "step-section" in prompt
    assert "hist-should-drop" not in prompt
    assert len(prompt) > config.prompt_char_budget


def test_build_prompt_truncates_fixed_section_budgets() -> None:
    config = AgentConfig(prompt_char_budget=10000)
    comp = ContextCompressor(config)
    system = "S" * 500
    task = "T" * 500
    state = "A" * 800
    step_info = "P" * 600
    prompt = comp.build_prompt(
        system=system,
        task=task,
        state=state,
        step_info=step_info,
        history=[],
    )
    # 各区段按预算截断后拼装
    expected = f"{system[:300]}\n\n{task[:200]}\n\n{state[:500]}\n\n{step_info[:400]}"
    assert prompt == expected


def test_summarize_state_format_and_keep_last(compressor: ContextCompressor) -> None:
    history = ["一步", "二步", "三步", "四步"]
    text = compressor.summarize_state(history, keep_last=3)
    assert text.startswith("已完成 4 步：")
    assert "二步" in text and "三步" in text and "四步" in text
    assert "一步" not in text


def test_summarize_state_empty(compressor: ContextCompressor) -> None:
    assert compressor.summarize_state([]) == "已完成 0 步："


def test_summarize_state_truncates_over_500(compressor: ContextCompressor) -> None:
    history = ["X" * 300, "Y" * 300]
    text = compressor.summarize_state(history, keep_last=2)
    assert len(text) <= 500
    assert text.startswith("已完成 2 步：")
