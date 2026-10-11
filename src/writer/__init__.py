from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import (
    LLMClientProtocol,
    LLMFormatError,
    apply_max_tokens,
    strip_think_blocks,
    take_thinking,
)
from pocket_agent.prompts import OUTLINE, OUTLINE_SCHEMA, SECTION_WRITE
from pocket_agent.task import parse_total_chars as _parse_total_chars

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
# 段长不足目标此比例时触发续写（最多一轮，仍用 SECTION_WRITE）
_SHORT_SECTION_RATIO = 0.60
_MAX_SECTIONS = 100
_DEFAULT_TOTAL_CHARS = 3000
# 单段允许略超目标；全文硬顶
_SECTION_MAX_RATIO = 1.05
_ARTICLE_MAX_RATIO = 1.08
_ARTICLE_MIN_RATIO = 0.90
_MIN_TRIM_CHARS = 4
_SEAM_TARGET = 0.05


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

    def seam_ngram_overlap(
        self,
        text_a: str,
        text_b: str,
        n: int = 3,
        window: int = 80,
    ) -> float:
        """段间接缝窗口内的 3-gram 重叠率（评测用，避免同主题全文误伤）。"""
        if window <= 0:
            return 0.0
        return self.ngram_overlap(text_a[-window:], text_b[:window], n=n)

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
        self.on_thinking: Callable[[str], None] | None = None
        self.last_outline: list[SectionSpec] = []
        self.last_sections: list[str] = []
        self.last_card: ConsistencyCard = ConsistencyCard()
        self.used_outline_fallback: bool = False

    def write(
        self,
        topic: str,
        total_chars: int,
        material: str = "",
        plan: list[str] | None = None,
    ) -> str:
        """四段式流水线：大纲 → 一致性卡 → 逐段撰写 → 拼接去重。"""
        outline = self._build_outline(topic, total_chars, material, plan=plan)
        self.last_outline = outline
        card = self._build_consistency_card(material, outline)
        self.last_card = card

        outline_text = _format_outline(outline)
        consistency_text = _serialize_facts(card.facts) or "（无）"
        written: list[str] = []
        n_sections = len(outline)
        hard_cap = int(total_chars * _ARTICLE_MAX_RATIO)

        for index, section in enumerate(outline, start=1):
            used = sum(len(s) for s in written)
            if used >= hard_cap:
                break
            remain_sections = n_sections - index + 1
            # 按剩余预算动态下调本段目标，抑制超写
            budget = max(80, (total_chars - used) // max(remain_sections, 1))
            section.target_chars = min(section.target_chars, budget)

            body = self._write_section(
                topic=topic,
                outline_text=outline_text,
                consistency_text=consistency_text,
                index=index,
                section=section,
                previous=written[-1] if written else "",
            )
            if written:
                body = trim_overlap(
                    written[-1],
                    body,
                    self._config.overlap_max,
                    seam_search=self._config.seam_tail_chars,
                )
            body = self._expand_if_short(
                topic=topic,
                outline_text=outline_text,
                consistency_text=consistency_text,
                index=index,
                section=section,
                body=body,
            )
            if written:
                body = trim_overlap(
                    written[-1],
                    body,
                    self._config.overlap_max,
                    seam_search=self._config.seam_tail_chars,
                )
            body = clip_to_target(body, section.target_chars, _SECTION_MAX_RATIO)
            # 累计将破硬顶时，本段再裁一刀
            room = hard_cap - used
            if room <= 0:
                break
            if len(body) > room:
                body = clip_to_target(body, room, 1.0)
            if not body.strip():
                continue
            written.append(body)
            if self.on_section is not None:
                self.on_section(index, section, body)

        self.last_sections = written
        article = "\n\n".join(written)
        article = self._pad_to_total(
            topic=topic,
            article=article,
            written=written,
            outline_text=outline_text,
            consistency_text=consistency_text,
            total_chars=total_chars,
        )
        # 超标则从文末按句裁到目标上限
        if len(article) > hard_cap:
            article = clip_to_target(article, total_chars, _ARTICLE_MAX_RATIO)
            # 同步裁尾部段落列表，避免指标与正文不一致
            self.last_sections = _resync_sections_to_article(written, article)
        return article

    def _write_section(
        self,
        *,
        topic: str,
        outline_text: str,
        consistency_text: str,
        index: int,
        section: SectionSpec,
        previous: str,
        extra_points: list[str] | None = None,
    ) -> str:
        seam_text = (
            previous[-self._config.seam_tail_chars :] if previous else "（无）"
        )
        points = list(section.points)
        if extra_points:
            points = [*points, *extra_points]
        points_text = "；".join(points) if points else "（无）"
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
        saved_tokens = _current_max_tokens(self._llm)
        apply_max_tokens(self._llm, section_max_tokens(section.target_chars))
        try:
            raw = str(self._llm.generate(prompt)).strip()
        finally:
            if saved_tokens is not None:
                apply_max_tokens(self._llm, saved_tokens)
        thinking = take_thinking(self._llm)
        if thinking and self._config.reasoning_enabled and self.on_thinking is not None:
            self.on_thinking(thinking)
        return _strip_section_echo(strip_think_blocks(raw))

    def _expand_if_short(
        self,
        *,
        topic: str,
        outline_text: str,
        consistency_text: str,
        index: int,
        section: SectionSpec,
        body: str,
    ) -> str:
        need = int(section.target_chars * _SHORT_SECTION_RATIO)
        # 正文过短视为 stub/失败，避免单测 Mock 队列被续写抽干
        if section.target_chars < 200 or len(body) < 80 or len(body) >= need:
            return body
        remain = max(80, section.target_chars - len(body))
        more = self._write_section(
            topic=topic,
            outline_text=outline_text,
            consistency_text=consistency_text,
            index=index,
            section=section,
            previous=body,
            extra_points=[
                f"在已有正文基础上续写约{remain}字，补足本段目标字数，禁止重复已写句子",
            ],
        )
        more = trim_overlap(
            body,
            more,
            self._config.overlap_max,
            seam_search=self._config.seam_tail_chars,
        )
        if not more.strip():
            return body
        merged = f"{body.rstrip()}\n{more.lstrip()}"
        return clip_to_target(merged, section.target_chars, _SECTION_MAX_RATIO)

    def _pad_to_total(
        self,
        *,
        topic: str,
        article: str,
        written: list[str],
        outline_text: str,
        consistency_text: str,
        total_chars: int,
    ) -> str:
        """全文仍不足目标 85% 时，按剩余预算追加补写段（不突破硬顶）。"""
        floor = int(total_chars * _ARTICLE_MIN_RATIO)
        hard_cap = int(total_chars * _ARTICLE_MAX_RATIO)
        if len(article) >= floor or not written:
            return article
        avg = len(article) / len(written)
        if avg < self._config.section_char_target * 0.3:
            return article

        chunk = max(200, self._config.section_char_target)
        guard = 0
        while len(article) < floor and len(article) < hard_cap and guard < 40:
            remain = hard_cap - len(article)
            if remain < 80:
                break
            target = min(chunk, remain, max(80, total_chars - len(article)))
            index = len(written) + 1
            section = SectionSpec(
                title=f"补充第{index}部分",
                points=["围绕主题继续展开，勿重复前文论点"],
                target_chars=target,
            )
            body = self._write_section(
                topic=topic,
                outline_text=outline_text,
                consistency_text=consistency_text,
                index=index,
                section=section,
                previous=written[-1] if written else "",
            )
            if written:
                body = trim_overlap(
                    written[-1],
                    body,
                    self._config.overlap_max,
                    seam_search=self._config.seam_tail_chars,
                )
            body = clip_to_target(body, target, _SECTION_MAX_RATIO)
            if len(body) > remain:
                body = clip_to_target(body, remain, 1.0)
            if not body.strip():
                break
            written.append(body)
            self.last_outline.append(section)
            if self.on_section is not None:
                self.on_section(index, section, body)
            article = "\n\n".join(written)
            guard += 1
        self.last_sections = written
        return article

    def _build_outline(
        self,
        topic: str,
        total_chars: int,
        material: str,
        plan: list[str] | None = None,
    ) -> list[SectionSpec]:
        self.used_outline_fallback = False
        desired_n = _desired_section_count(total_chars, self._config.section_char_target)
        if plan:
            parsed: list[SectionSpec] = []
            for i, step in enumerate(plan, start=1):
                title = str(step).strip() or f"第{i}部分"
                parsed.append(SectionSpec(title=title[:40], points=[], target_chars=0))
            if parsed:
                parsed = _ensure_section_count(parsed, desired_n)
                n = len(parsed)
                base = max(1, total_chars // n)
                for section in parsed:
                    section.target_chars = base
                return parsed
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

        # 段数过少会导致单段目标过大、模型写不满；补齐到 desired_n
        parsed = _ensure_section_count(parsed, desired_n)
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


def trim_overlap(
    previous: str,
    current: str,
    overlap_max: int,
    *,
    seam_search: int | None = None,
) -> str:
    """裁掉段首与前段的复述：精确前后缀、尾窗子串、整句，多轮直到接缝重叠达标。"""
    if not previous or not current or overlap_max <= 0:
        return current

    search_win = seam_search if seam_search and seam_search > 0 else max(
        overlap_max * 3, 200
    )
    metrics = WriterMetrics()
    text = current

    for _ in range(48):
        before = text
        if not text:
            break

        # 1) 最长公共前后缀（可多轮，每轮最多 overlap_max）
        max_n = min(overlap_max, len(previous), len(text))
        best = 0
        for n in range(max_n, _MIN_TRIM_CHARS - 1, -1):
            if previous[-n:] == text[:n]:
                best = n
                break
        if best:
            text = text[best:].lstrip("\n　 ")
            continue

        # 2) 段首长前缀出现在前段尾窗内（覆盖「前段多了总之，」这类错位复述）
        tail = previous[-search_win:]
        best = 0
        upper = min(len(text), len(tail), max(overlap_max * 4, 320))
        for n in range(upper, _MIN_TRIM_CHARS - 1, -1):
            if text[:n] in tail:
                best = n
                break
        if best:
            text = text[best:].lstrip("\n　 ")
            continue

        # 3) 剥除段首整句（出现在前段）
        prev_sents = {
            s.strip() for s in _split_sentences(previous) if len(s.strip()) >= 8
        }
        head = next((s for s in _split_sentences(text) if s.strip()), "")
        stripped = head.strip()
        if stripped and stripped in prev_sents and text.lstrip().startswith(stripped):
            idx = text.find(stripped)
            if idx >= 0:
                text = text[idx + len(stripped) :].lstrip("\n　 ")
                continue

        # 4) 段首句与前段某句高度相似（或短前缀已出现）则剥除
        head = next((s for s in _split_sentences(text) if s.strip()), "")
        stripped = head.strip()
        if stripped and len(stripped) >= 8 and text.lstrip().startswith(stripped):
            similar = any(
                SequenceMatcher(None, stripped, ps).ratio() >= 0.72 for ps in prev_sents
            )
            prefix_hit = len(stripped) >= 16 and stripped[:16] in previous
            if similar or prefix_hit:
                idx = text.find(stripped)
                if idx >= 0:
                    text = text[idx + len(stripped) :].lstrip("\n　 ")
                    continue

        # 5) 接缝窗仍严重重叠：再剥一句（仅当重叠极高，避免误伤）
        if (
            text
            and metrics.seam_ngram_overlap(previous, text, n=3, window=overlap_max)
            >= 0.35
        ):
            head = next((s for s in _split_sentences(text) if s.strip()), "")
            if head.strip() and len(head.strip()) >= 8:
                idx = text.find(head.strip())
                if idx >= 0:
                    text = text[idx + len(head.strip()) :].lstrip("\n　 ")
                    if text != before:
                        continue

        if text == before:
            break

    return text


def clip_to_target(text: str, target_chars: int, max_ratio: float) -> str:
    """按句号边界裁到 target*max_ratio，避免单段/全文超写。"""
    if target_chars <= 0 or not text:
        return text
    limit = max(1, int(target_chars * max_ratio))
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [cut.rfind(ch) for ch in ("。", "！", "？", "\n")]
    pos = max(ends)
    if pos >= max(8, limit // 5):
        return cut[: pos + 1].rstrip()
    return cut.rstrip()


def _resync_sections_to_article(sections: list[str], article: str) -> list[str]:
    """文末裁剪后，丢弃已不在正文中的尾部段落。"""
    if not sections:
        return []
    kept: list[str] = []
    cursor = 0
    for section in sections:
        idx = article.find(section[: min(40, len(section))], cursor)
        if idx < 0:
            # 段落被裁短：保留 article 剩余作为末段
            rest = article[cursor:].strip()
            if rest:
                kept.append(rest)
            break
        # 若本段完整仍在
        if article[idx : idx + len(section)] == section:
            kept.append(section)
            cursor = idx + len(section)
        else:
            rest = article[idx:].strip()
            if rest:
                kept.append(rest)
            break
    return kept or ([article] if article else [])


def parse_total_chars(text: str, default: int = _DEFAULT_TOTAL_CHARS) -> int:
    """兼容入口：字数解析已上移到任务卡 / Planner。"""
    parsed = _parse_total_chars(text, default=0)
    return parsed if parsed > 0 else max(1, default)


def section_max_tokens(target_chars: int) -> int:
    """中文字数 × 1.5 + 冗余，保证单段不被全局 512 截断。"""
    return max(256, int(max(1, target_chars) * 1.5) + 64)


def _current_max_tokens(llm: object) -> int | None:
    cfg = getattr(llm, "_config", None)
    if cfg is not None and hasattr(cfg, "max_tokens"):
        return int(cfg.max_tokens)
    if hasattr(llm, "_max_tokens"):
        return int(llm._max_tokens)
    return None


def _desired_section_count(total_chars: int, section_chars: int) -> int:
    chunk = section_chars if section_chars > 0 else 500
    return min(_MAX_SECTIONS, max(1, math.ceil(total_chars / chunk)))


def _ensure_section_count(
    sections: list[SectionSpec],
    desired_n: int,
) -> list[SectionSpec]:
    desired_n = min(_MAX_SECTIONS, max(1, desired_n))
    if len(sections) >= desired_n:
        return sections[:desired_n]
    out = list(sections)
    while len(out) < desired_n:
        i = len(out) + 1
        out.append(
            SectionSpec(
                title=f"第{i}部分",
                points=["围绕主题继续展开，补充新论据与细节"],
                target_chars=0,
            )
        )
    return out


def _uniform_outline(total_chars: int, section_chars: int) -> list[SectionSpec]:
    n = _desired_section_count(total_chars, section_chars)
    target = max(1, total_chars // n)
    return [
        SectionSpec(title=f"第{i}部分", points=[], target_chars=target)
        for i in range(1, n + 1)
    ]


_SECTION_ECHO_LINE = re.compile(
    r"^\s*只输出(?:本段正文|JSON|摘要本身|标题本身|数字)[。.]?\s*$",
    re.MULTILINE,
)
_SECTION_ECHO_INLINE = re.compile(r"只输出本段正文。?")


def _strip_section_echo(text: str) -> str:
    """去掉段落生成回声的指令句（如「只输出本段正文。」）。"""
    cleaned = _SECTION_ECHO_LINE.sub("", text or "")
    cleaned = _SECTION_ECHO_INLINE.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _strip_thinking(text: str) -> str:
    """兼容入口：统一走 strip_think_blocks。"""
    return strip_think_blocks(text)


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
