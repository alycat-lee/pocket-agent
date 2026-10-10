from __future__ import annotations

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMFormatError
from pocket_agent.writer import (
    LongWriter,
    SectionSpec,
    WriterMetrics,
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


def test_metrics() -> None:
    metrics = WriterMetrics()

    # Jaccard：abcde vs cdefg，n=3 → {abc,bcd,cde} vs {cde,def,efg} → 交1并5 = 0.2
    assert metrics.ngram_overlap("abcde", "cdefg", n=3) == 0.2
    assert metrics.ngram_overlap("", "abc", n=3) == 0.0

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
