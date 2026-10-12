"""嵌入后端：hash stub 始终可用；本地/云端进 extras，缺失则降级。"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from typing import Any, Protocol

import httpx

DEFAULT_DIM = 64
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"


class Embedder(Protocol):
    name: str
    available: bool
    reason: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """确定性字符 2-gram 哈希向量，供测试与无 extras 时的流水线验证。"""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = max(8, int(dim))
        self.name = "hash"
        self.available = True
        self.reason = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        raw = text or ""
        if not raw:
            vec[0] = 1.0
            return vec
        for index in range(len(raw)):
            gram = raw[index : index + 2]
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:2], "little") % self.dim
            vec[slot] += 1.0
        return _l2_normalize(vec)


class UnavailableEmbedder:
    def __init__(self, reason: str, name: str = "unavailable") -> None:
        self.name = name
        self.available = False
        self.reason = reason

    def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise RuntimeError(self.reason or "embedding 不可用")


class LocalEmbedder:
    """sentence-transformers 本地模型，强制 CPU。延迟导入 extras。"""

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL) -> None:
        self.name = "local"
        self.available = False
        self.reason = ""
        self._model_name = model
        self._model: object | None = None
        # 可选依赖：未安装时降级，禁止在模块顶层导入。
        if importlib.util.find_spec("sentence_transformers") is None:
            self.reason = (
                "未安装 sentence-transformers。"
                "请执行 pip install 'pocket-agent[knowledge]'"
            )
            return
        self.available = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(texts, convert_to_numpy=False)
        out: list[list[float]] = []
        for row in vectors:
            out.append([float(x) for x in list(row)])
        return out

    def _ensure_model(self) -> Any:
        if not self.available:
            raise RuntimeError(self.reason or "本地 embedding 不可用")
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self._model_name, device="cpu")
            except Exception as exc:
                self.available = False
                self.reason = f"本地 embedding 加载失败：{exc}"
                raise RuntimeError(self.reason) from exc
        return self._model


class CloudEmbedder:
    """OpenAI 兼容 /embeddings。"""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
    ) -> None:
        self.name = "cloud"
        self.available = bool((base_url or "").strip())
        self.reason = "" if self.available else "未配置 embedding base_url"
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise RuntimeError(self.reason)

        url = f"{self._base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        with httpx.Client(timeout=self._timeout, trust_env=False) as client:
            res = client.post(
                url,
                headers=headers,
                json={"model": self._model, "input": texts},
            )
            res.raise_for_status()
            payload = res.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise TypeError("embedding 响应缺少 data")
        out: list[list[float]] = []
        for item in rows:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list):
                raise TypeError("embedding 向量格式错误")
            out.append([float(x) for x in vec])
        return out


class LexicalEmbedder:
    """去停用 bigram + IDF 覆盖率检索；ingest 仍写占位向量以保持 sqlite 表结构。"""

    def __init__(self, *, alias: str = "lexical") -> None:
        self.name = alias if alias in {"lexical", "hash"} else "lexical"
        self.available = True
        self.reason = ""
        self._hash = HashEmbedder()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._hash.embed(texts)


def resolve_embedder(kind: str, config: dict[str, object]) -> Embedder:
    name = (kind or "lexical").strip().lower()
    if name in {"hash", "lexical"}:
        return LexicalEmbedder(alias=name)
    if name == "cloud":
        return CloudEmbedder(
            str(config.get("embed_base_url") or config.get("base_url") or ""),
            api_key=str(config.get("embed_api_key") or config.get("api_key") or ""),
            model=str(config.get("embed_model") or "text-embedding-3-small"),
        )
    if name == "unavailable":
        return UnavailableEmbedder(str(config.get("reason") or "embedding 已禁用"))
    return LocalEmbedder(str(config.get("embed_model") or DEFAULT_LOCAL_MODEL))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(n))
    return dot / (_norm(left) * _norm(right))


def _l2_normalize(vec: list[float]) -> list[float]:
    n = _norm(vec)
    return [x / n for x in vec]


def _norm(vec: list[float]) -> float:
    value = math.sqrt(sum(x * x for x in vec))
    return value if value > 0 else 1.0
