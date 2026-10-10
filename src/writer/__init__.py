from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMClientProtocol, LLMFormatError
from pocket_agent.prompts import OUTLINE, OUTLINE_SCHEMA, SECTION_WRITE

_SENT_SPLIT = re.compile(r"(?<=[。！？\n])")
_FACT_NUM_SENT = re.compile(r"\d")
# 先定位「为/是/达… + 数字」，再取连接词前 2~3 个汉字（或英文词）作指标名
_METRIC_ANCHOR_RE = re.compile(
    r"([为是达至：:])\s*(-?\d+(?:\.\d+)?)\s*([%％万亿元]*)"
)
_LABEL_BEFORE_RE = re.compile(
    r"(?:[\u4e00-\u9fff]{2,3}|[A-Za-z][A-Za-z0-9_\-]{1,24})$"
)
_CONSISTENCY_CHAR_LIMIT = 300


@dataclass
class SectionSpec:
    title: str
    points: list[str]  # 本段要点
    target_chars: int


@dataclass
class ConsistencyCard:
    facts: list[str] = field(default_factory=list)  # 序列化后 ≤ 300 字符


class WriterMetrics:
    """自动连贯性指标，供测试与评测调用。"""

    def ngram_overlap(self, text_a: str, text_b: str, n: int = 3) -> float:
        """字符 n-gram Jaccard 重叠率。"""
        if n < 1 or not text_a or not text_b:
            return 0.0
        ga = _char_ngrams(text_a, n)
        gb = _char_ngrams(text_b, n)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)

    def numeric_consistency(self, full_text: str) -> list[str]:
        """抽取全文数字，返回同一指标多处出现但取值不一致的冲突列表。"""
        grouped: dict[str, set[str]] = {}
        for match in _METRIC_ANCHOR_RE.finditer(full_text):
            label_match = _LABEL_BEFORE_RE.search(full_text[: match.start(1)])
            if label_match is None:
                continue
            label = label_match.group(0)
            value = match.group(2)
            unit = match.group(3) or ""
            key = f"{label}{unit}"
            grouped.setdefault(key, set()).add(value)
        conflicts: list[str] = []
        for key, values in sorted(grouped.items()):
            if len(values) > 1:
                conflicts.append(f"{key}: {', '.join(sorted(values, key=_num_sort_key))}")
        return conflicts

    def outline_coverage(
        self,
        outline: list[SectionSpec],
        full_text: str,
    ) -> float:
        if not outline:
            return 1.0
        hits = 0
        for section in outline:
            markers = [section.title, *section.points]
            markers = [m.strip() for m in markers if m and m.strip()]
            if not markers or any(m in full_text for m in markers):
                hits += 1
        return hits / len(outline)


class LongWriter:
    def __init__(self, config: AgentConfig, llm: LLMClientProtocol) -> None:
        self._config = config
        self._llm = llm
        self.on_section: Callable[[int, SectionSpec, str], None] | None = None
        self.last_outline: list[SectionSpec] = []
        self.last_sections: list[str] = []
        self.last_card: ConsistencyCard = ConsistencyCard()
        self.used_outline_fallback: bool = False

    def write(self, topic: str, total_chars: int, material: str = "") -> str:
        """四段式流水线：大纲 → 一致性卡 → 逐段撰写 → 拼接去重。"""
        outline = self._build_outline(topic, total_chars, material)
        self.last_outline = outline
        card = self._build_consistency_card(material, outline)
        self.last_card = card

        outline_text = _format_outline(outline)
        consistency_text = _serialize_facts(card.facts) or "（无）"
        written: list[str] = []

        for index, section in enumerate(outline, start=1):
            seam_text = (
                written[-1][-self._config.seam_tail_chars :]
                if written
                else "（无）"
            )
            points_text = "；".join(section.points) if section.points else "（无）"
            prompt = SECTION_WRITE.format(
                topic=topic,
                outline_text=outline_text,
                consistency_text=consistency_text,
                index=index,
                section_title=section.title,
                points=points_text,
                target_chars=section.target_chars,
                seam_text=seam_text,
            )
            body = str(self._llm.generate(prompt)).strip()
            if written:
                body = trim_overlap(written[-1], body, self._config.overlap_max)
            written.append(body)
            if self.on_section is not None:
                self.on_section(index, section, body)

        self.last_sections = written
        return "\n\n".join(written)

    def _build_outline(
        self,
        topic: str,
        total_chars: int,
        material: str,
    ) -> list[SectionSpec]:
        self.used_outline_fallback = False
        material_text = material.strip() if material.strip() else "（无）"
        prompt = OUTLINE.format(
            topic=topic,
            total_chars=total_chars,
            section_chars=self._config.section_char_target,
            material=material_text,
        )
        try:
            data = self._llm.generate_json(prompt, OUTLINE_SCHEMA)
            raw_sections = data.get("sections", [])
            if not isinstance(raw_sections, list) or not raw_sections:
                raise LLMFormatError("outline sections empty")
            parsed: list[SectionSpec] = []
            for item in raw_sections:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip() or "未命名段落"
                points_raw = item.get("points", [])
                points: list[str] = []
                if isinstance(points_raw, list):
                    points = [str(p).strip() for p in points_raw if str(p).strip()]
                parsed.append(SectionSpec(title=title, points=points, target_chars=0))
            if not parsed:
                raise LLMFormatError("outline parse yielded no sections")
        except LLMFormatError:
            self.used_outline_fallback = True
            return _uniform_outline(total_chars, self._config.section_char_target)

        n = len(parsed)
        base = max(1, total_chars // n)
        for section in parsed:
            section.target_chars = base
        return parsed

    def _build_consistency_card(
        self,
        material: str,
        outline: list[SectionSpec],
    ) -> ConsistencyCard:
        if material.strip():
            facts = _extract_facts_from_material(material)
        else:
            facts = _extract_facts_from_outline(outline)
        return ConsistencyCard(facts=_fit_facts(facts, _CONSISTENCY_CHAR_LIMIT))


def trim_overlap(previous: str, current: str, overlap_max: int) -> str:
    """裁掉段首与前段结尾的重叠（子串优先，其次整句），窗口 ≤ overlap_max。"""
    if not previous or not current or overlap_max <= 0:
        return current

    max_n = min(overlap_max, len(previous), len(current))
    for n in range(max_n, 0, -1):
        if previous[-n:] == current[:n]:
            return current[n:].lstrip("\n")

    tail = previous[-overlap_max:]
    for sent in _split_sentences(current):
        stripped = sent.strip()
        if not stripped:
            continue
        if stripped in tail and current.lstrip().startswith(stripped):
            idx = current.find(stripped)
            if idx >= 0:
                return current[idx + len(stripped) :].lstrip("\n")
        break
    return current


def _uniform_outline(total_chars: int, section_chars: int) -> list[SectionSpec]:
    chunk = section_chars if section_chars > 0 else 500
    n = max(1, int(total_chars / chunk))
    target = max(1, total_chars // n)
    return [
        SectionSpec(title=f"第{i}部分", points=[], target_chars=target)
        for i in range(1, n + 1)
    ]


def _extract_facts_from_material(material: str) -> list[str]:
    facts: list[str] = []
    lines = [ln.strip() for ln in material.splitlines() if ln.strip()]
    if lines:
        facts.append(lines[0][:80])
    for sent in re.split(r"[。！？\n]", material):
        sent = sent.strip()
        if sent and _FACT_NUM_SENT.search(sent):
            facts.append(sent)
    return _dedupe(facts)


def _extract_facts_from_outline(outline: list[SectionSpec]) -> list[str]:
    facts: list[str] = []
    for section in outline:
        if section.title.strip():
            facts.append(section.title.strip())
        facts.extend(p.strip() for p in section.points if p.strip())
    return _dedupe(facts)


def _fit_facts(facts: list[str], limit: int) -> list[str]:
    kept: list[str] = []
    size = 0
    for fact in facts:
        piece = fact.strip()
        if not piece:
            continue
        add = len(piece) + (1 if kept else 0)
        if size + add > limit:
            remain = limit - size - (1 if kept else 0)
            if remain > 8:
                kept.append(piece[:remain])
            break
        kept.append(piece)
        size += add
    return kept


def _serialize_facts(facts: list[str]) -> str:
    return "；".join(facts)


def _format_outline(outline: list[SectionSpec]) -> str:
    lines: list[str] = []
    for i, section in enumerate(outline, start=1):
        points = "；".join(section.points) if section.points else "（无要点）"
        lines.append(f"{i}. {section.title}（约{section.target_chars}字）：{points}")
    return "\n".join(lines)


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_SPLIT.split(text) if p]
    return parts if parts else [text]


def _char_ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _num_sort_key(value: str) -> tuple[float, str]:
    try:
        return (float(value), value)
    except ValueError:
        return (0.0, value)
