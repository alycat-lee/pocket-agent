from __future__ import annotations

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMFormatError
from pocket_agent.writer import (
    LongWriter,
    SectionSpec,
    WriterMetrics,
    _desired_section_count,
    _strip_section_echo,
    clip_to_target,
    parse_total_chars,
    section_max_tokens,
    trim_overlap,
)


def test_pipeline_order(mock_llm_factory: type) -> None:
    outline = {
        "sections": [
            {"title": "背景概述", "points": ["定义范围", "研究意义"]},
            {"title": "现状分析", "points": ["市场规模"]},
            {"title": "未来展望", "points": ["技术路线", "政策趋势"]},
        ]
    }
    llm = mock_llm_factory(
        [
            outline,
            "背景段正文，覆盖定义范围。",
            "现状段正文，覆盖市场规模。",
            "展望段正文，覆盖技术路线。",
        ]
    )
    writer = LongWriter(AgentConfig(max_tokens=256), llm)
    text = writer.write("新能源汽车行业分析", total_chars=1500)

    assert len(writer.last_outline) == 3
    assert len(writer.last_sections) == 3
    assert "背景段正文" in text
    assert llm.prompts[0].startswith("为主题「新能源汽车行业分析」")
    assert "只输出 JSON" in llm.prompts[0]
    assert all("你正在逐段撰写文章" in p for p in llm.prompts[1:])
    assert "第 1 段「背景概述」" in llm.prompts[1]
    assert "第 2 段「现状分析」" in llm.prompts[2]
    assert "第 3 段「未来展望」" in llm.prompts[3]


def test_overlap_trimming(mock_llm_factory: type) -> None:
    outline = {
        "sections": [
            {"title": "第一段", "points": ["a"]},
            {"title": "第二段", "points": ["b"]},
            {"title": "第三段", "points": ["c"]},
        ]
    }
    llm = mock_llm_factory(
        [
            outline,
            "开篇独特内容。共享尾巴句。",
            "共享尾巴句。第二段独有推进。",
            "第二段独有推进。第三段收束结论。",
        ]
    )
    writer = LongWriter(AgentConfig(overlap_max=80), llm)
    text = writer.write("重叠测试", total_chars=1200)

    assert text.count("共享尾巴句。") == 1
    assert text.count("第二段独有推进。") == 1
    assert "第三段收束结论。" in text
    # 单元级：裁剪函数本身
    assert trim_overlap("前文。重叠句。", "重叠句。后文。", 80) == "后文。"

    # 错位复述：前段多「总之，」，后段整段抄尾部
    para = (
        "在测试管理中，工具的选择与应用是关键环节。"
        "通过合适的工具可以有效降低缺陷率，确保产品质量。"
    )
    prev = "总之，" + para
    curr = para + "本段独有收束。"
    trimmed = trim_overlap(prev, curr, 80, seam_search=250)
    assert "工具的选择与应用是关键环节" not in trimmed
    assert "本段独有收束" in trimmed
    assert WriterMetrics().seam_ngram_overlap(prev, trimmed, n=3, window=80) < 0.05


def test_clip_to_target_caps_overshoot() -> None:
    text = "第一句完。第二句完。第三句完。第四句完。第五句完。"
    clipped = clip_to_target(text, target_chars=20, max_ratio=1.05)
    assert len(clipped) <= int(20 * 1.05) + 1
    assert clipped.endswith("。")


def test_outline_fallback(mock_llm_factory: type) -> None:
    llm = mock_llm_factory(
        [
            LLMFormatError("outline broken"),
            "兜底段一正文。",
            "兜底段二正文。",
            "兜底段三正文。",
        ]
    )
    # total_chars/500 = 3
    writer = LongWriter(AgentConfig(section_char_target=500), llm)
    text = writer.write("兜底主题", total_chars=1500)

    assert writer.used_outline_fallback is True
    assert len(writer.last_outline) == 3
    assert all(s.points == [] for s in writer.last_outline)
    assert [s.title for s in writer.last_outline] == ["第1部分", "第2部分", "第3部分"]
    assert "兜底段一正文" in text
    assert "兜底段三正文" in text


def test_outline_padded_when_too_few_sections(mock_llm_factory: type) -> None:
    """LLM 只给 2 段时，应按 total/section_char_target 补齐段数。"""
    outline = {
        "sections": [
            {"title": "只有一段", "points": ["a"]},
            {"title": "只有二段", "points": ["b"]},
        ]
    }
    # 1500/500=3 → 需补第3段
    llm = mock_llm_factory(
        [
            outline,
            "第一段正文内容。",
            "第二段正文内容。",
            "第三段正文内容。",
        ]
    )
    writer = LongWriter(AgentConfig(section_char_target=500), llm)
    writer.write("补齐测试", total_chars=1500)
    assert len(writer.last_outline) == 3
    assert writer.last_outline[2].title == "第3部分"
    assert all(s.target_chars == 500 for s in writer.last_outline)


def test_metrics() -> None:
    metrics = WriterMetrics()

    # Jaccard：abcde vs cdefg，n=3 → {abc,bcd,cde} vs {cde,def,efg} → 交1并5 = 0.2
    assert metrics.ngram_overlap("abcde", "cdefg", n=3) == 0.2
    assert metrics.ngram_overlap("", "abc", n=3) == 0.0

    # 接缝窗：去重后窗内重叠应显著低于去重前
    a = ("甲" * 100) + "接缝句唯一。"
    b = "接缝句唯一。" + ("乙" * 100)
    before = metrics.seam_ngram_overlap(a, b, n=3, window=80)
    b2 = trim_overlap(a, b, 80)
    assert not b2.startswith("接缝句唯一")
    after = metrics.seam_ngram_overlap(a, b2, n=3, window=80)
    assert before > 0.05
    assert after < before
    assert after < 0.05

    conflicts = metrics.numeric_consistency(
        "利润率为15%。营收为100万。另一处利润率为18%。营收为100万。"
    )
    assert any("利润率" in c and "15" in c and "18" in c for c in conflicts)
    assert not any("营收" in c for c in conflicts)

    outline = [
        SectionSpec(title="背景概述", points=["定义范围"], target_chars=100),
        SectionSpec(title="缺失标题XYZ", points=["从未出现的要点QQQ"], target_chars=100),
    ]
    full = "本文先讲背景概述，并说明定义范围，然后展开其他内容。"
    coverage = metrics.outline_coverage(outline, full)
    assert coverage == 0.5


def test_parse_total_chars_wanzi() -> None:
    assert parse_total_chars("给我生成一篇2万字的小说 内容随意") == 20000
    assert parse_total_chars("写20000字报告") == 20000
    assert parse_total_chars("写 2.5 万字") == 25000
    assert parse_total_chars("随便写一篇") == 3000


def test_desired_section_count_independent_and_capped() -> None:
    assert _desired_section_count(20000, 500) == 40
    assert _desired_section_count(1_000_000, 500) == 100


def test_short_section_continues_once(mock_llm_factory: type) -> None:
    outline = {"sections": [{"title": "一段", "points": ["展开"]}]}
    short = "甲" * 100
    more = "乙" * 80
    llm = mock_llm_factory([outline, short, more])
    writer = LongWriter(AgentConfig(section_char_target=500), llm)
    text = writer.write("续写测", total_chars=200)
    assert "甲" in text
    assert "乙" in text
    assert sum(1 for p in llm.prompts if "你正在逐段撰写文章" in p) == 2


def test_section_max_tokens_scales_with_target(mock_llm_factory: type) -> None:
    outline = {"sections": [{"title": "一段", "points": ["a"]}]}
    inner = mock_llm_factory([outline, "正" * 460])

    class _Spy:
        def __init__(self) -> None:
            self._config = AgentConfig(max_tokens=512)
            self.seen: list[int] = []

        def generate(self, prompt: str) -> str:
            self.seen.append(self._config.max_tokens)
            return inner.generate(prompt)

        def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
            return inner.generate_int(prompt, minimum, maximum)

        def generate_json(self, prompt: str, schema: dict) -> dict:
            return inner.generate_json(prompt, schema)

    spy = _Spy()
    writer = LongWriter(AgentConfig(section_char_target=500, max_tokens=512), spy)
    writer.write("字数", total_chars=500)
    assert spy.seen
    assert max(spy.seen) == section_max_tokens(500)
    assert spy._config.max_tokens == 512


def test_strip_section_echo_drops_instruction_line() -> None:
    raw = "咖啡馆里很安静。\n\n只输出本段正文。\n\n他推门进来。"
    cleaned = _strip_section_echo(raw)
    assert "只输出本段正文" not in cleaned
    assert "咖啡馆里很安静。" in cleaned
    assert "他推门进来。" in cleaned
