"""知识库：切分 / embedding / sqlite 向量存储 / 检索压缩。"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pocket_agent.knowledge.embed import (
    Embedder,
    HashEmbedder,
    UnavailableEmbedder,
    cosine,
    resolve_embedder,
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMPTY_RESULT = "未找到相关资料"
HONEST_PHRASES = ("未找到相关资料", "未找到资料")
MIN_SCORE = 0.15


@dataclass
class KnowledgeHit:
    text: str
    source: str
    score: float
    doc_id: str
    chunk_id: str


@dataclass
class KnowledgeDocument:
    id: str
    name: str
    source: str
    chunks: int
    created_at: float


def split_text(
    text: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """按字切分；相邻块重叠 overlap 字。"""
    body = (text or "").strip()
    if not body:
        return []
    if size <= 0:
        return [body]
    safe_overlap = min(max(0, overlap), max(0, size - 1))
    if len(body) <= size:
        return [body]
    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + size, len(body))
        chunks.append(body[start:end])
        if end >= len(body):
            break
        start = end - safe_overlap
    return chunks


def parse_document(name: str, data: bytes) -> str:
    """txt/md 直接解码；pdf 尝试 pypdf（可选）。"""
    filename = (name or "").lower()
    if filename.endswith(".pdf") or data[:4] == b"%PDF":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("无法解析 PDF：未安装 pypdf") from exc
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("PDF 无可提取文本")
        return text
    return data.decode("utf-8", errors="replace")


def compress_hit(text: str, limit: int) -> str:
    body = " ".join((text or "").split())
    if limit <= 0 or len(body) <= limit:
        return body
    return body[: max(0, limit - 1)] + "…"


def format_hits(hits: list[KnowledgeHit], *, limit: int = 300) -> str:
    if not hits:
        return EMPTY_RESULT
    lines = [f"[来源：{hit.source}] {compress_hit(hit.text, limit)}" for hit in hits]
    return "\n".join(lines)


def attach_sources(answer: str, sources: list[str]) -> str:
    """代码拼装来源列表，不依赖模型自觉。"""
    unique: list[str] = []
    for item in sources:
        name = str(item).strip()
        if name and name not in unique:
            unique.append(name)
    body = (answer or "").rstrip()
    if not unique:
        return body
    if "\n来源：" in body or body.endswith("来源："):
        return body
    listing = "\n".join(f"- {name}" for name in unique)
    return f"{body}\n\n来源：\n{listing}"


def extract_sources(evidence: str) -> list[str]:
    names: list[str] = []
    for line in (evidence or "").splitlines():
        if line.startswith("[来源：") and "]" in line:
            names.append(line[len("[来源：") : line.index("]")])
    return names


class KnowledgeBase:
    """sqlite + 余弦相似度；embedder 不可用时降级，不崩溃。"""

    def __init__(
        self,
        db_path: str | Path,
        embedder: Embedder,
        *,
        top_k: int = 5,
        result_limit: int = 300,
    ) -> None:
        self._path = Path(db_path)
        self._embedder = embedder
        self.top_k = max(1, int(top_k))
        self.result_limit = max(40, int(result_limit))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> KnowledgeBase:
        data = dict(config or {})
        embedder = resolve_embedder(str(data.get("embedder") or "local"), data)
        db_path = data.get("db_path") or ".knowledge/kb.sqlite"
        return cls(
            Path(str(db_path)),
            embedder,
            top_k=int(data.get("top_k") or 5),
            result_limit=int(data.get("result_limit") or 300),
        )

    @property
    def available(self) -> bool:
        return bool(self._embedder.available)

    @property
    def reason(self) -> str:
        return str(getattr(self._embedder, "reason", "") or "")

    @property
    def embedder_name(self) -> str:
        return str(getattr(self._embedder, "name", "unknown"))

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "embedder": self.embedder_name,
            "doc_count": len(self.list_documents()),
        }

    def ingest(
        self,
        name: str,
        text: str,
        *,
        source: str = "",
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.reason or "知识库未启用：embedding 不可用")
        chunks = split_text(text)
        if not chunks:
            raise ValueError("文档为空")
        pid = doc_id or uuid.uuid4().hex[:12]
        vectors = self._embedder.embed(chunks)
        src = source or name
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents(id, name, source, created_at) VALUES (?,?,?,?)",
                (pid, name, src, now),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (pid,))
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                conn.execute(
                    "INSERT INTO chunks(id, doc_id, ordinal, text, embedding) VALUES (?,?,?,?,?)",
                    (
                        f"{pid}-{index}",
                        pid,
                        index,
                        chunk,
                        json.dumps(vector),
                    ),
                )
            conn.commit()
        return {"id": pid, "name": name, "source": src, "chunks": len(chunks)}

    def ingest_bytes(self, name: str, data: bytes, *, source: str = "") -> dict[str, Any]:
        return self.ingest(name, parse_document(name, data), source=source or name)

    def list_documents(self) -> list[KnowledgeDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT d.id, d.name, d.source, d.created_at, COUNT(c.id)
                FROM documents d
                LEFT JOIN chunks c ON c.doc_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [
            KnowledgeDocument(
                id=str(row[0]),
                name=str(row[1]),
                source=str(row[2]),
                created_at=float(row[3] or 0),
                chunks=int(row[4] or 0),
            )
            for row in rows
        ]

    def delete(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.commit()
            return cur.rowcount > 0

    def search(self, query: str, *, top_k: int | None = None) -> list[KnowledgeHit]:
        if not self.available:
            return []
        q = (query or "").strip()
        if not q:
            return []
        qvec = self._embedder.embed([q])[0]
        limit = top_k or self.top_k
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.doc_id, c.text, d.name, d.source, c.embedding
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                """
            ).fetchall()
        scored: list[KnowledgeHit] = []
        for row in rows:
            try:
                vector = [float(x) for x in json.loads(str(row[5]))]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            score = cosine(qvec, vector)
            source = str(row[4] or row[3] or row[1])
            scored.append(
                KnowledgeHit(
                    text=str(row[2]),
                    source=source,
                    score=score,
                    doc_id=str(row[1]),
                    chunk_id=str(row[0]),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return [item for item in scored[:limit] if item.score >= MIN_SCORE]

    def search_text(self, query: str, *, top_k: int | None = None) -> str:
        if not self.available:
            return f"{EMPTY_RESULT}（{self.reason or '知识库未启用'}）"
        return format_hits(self.search(query, top_k=top_k), limit=self.result_limit)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "EMPTY_RESULT",
    "HONEST_PHRASES",
    "HashEmbedder",
    "KnowledgeBase",
    "KnowledgeDocument",
    "KnowledgeHit",
    "UnavailableEmbedder",
    "attach_sources",
    "compress_hit",
    "extract_sources",
    "format_hits",
    "parse_document",
    "resolve_embedder",
    "split_text",
]
