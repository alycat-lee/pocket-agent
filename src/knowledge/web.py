"""外网检索：Wikipedia + SearXNG/Serper/DuckDuckGo，结果必须带来源标记。"""

from __future__ import annotations

import ipaddress
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from pocket_agent.knowledge import EMPTY_RESULT, KnowledgeHit, format_hits

WIKI_API = "https://zh.wikipedia.org/w/api.php"
DDG_HTML = "https://html.duckduckgo.com/html/"
DEFAULT_TIMEOUT = 8.0
_DDG_UA = "Mozilla/5.0"


def is_blocked_url(url: str) -> bool:
    """拦截非 http(s)、链路本地与云 metadata，防止 SearXNG 结果 SSRF。"""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True
    host = (parsed.hostname or "").lower()
    if not host:
        return True
    if host in {"metadata.google.internal", "metadata.google"}:
        return True
    if host == "169.254.169.254":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(addr.is_link_local or addr.is_multicast or addr.is_reserved)


def _is_blocked_url(url: str) -> bool:
    return is_blocked_url(url)


def wiki_lookup(query: str, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    q = (query or "").strip()
    if not q:
        return EMPTY_RESULT
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": "3",
        "format": "json",
        "utf8": "1",
    }
    try:
        with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=True) as client:
            res = client.get(WIKI_API, params=params)
            res.raise_for_status()
            payload = res.json()
    except (httpx.HTTPError, ValueError):
        return EMPTY_RESULT
    hits: list[KnowledgeHit] = []
    search = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
    for item in search if isinstance(search, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").replace("<span class=\"searchmatch\">", "")
        snippet = snippet.replace("</span>", "").strip()
        if not title:
            continue
        source = f"https://zh.wikipedia.org/wiki/{quote(title)}"
        hits.append(
            KnowledgeHit(
                text=snippet or title,
                source=source,
                score=1.0,
                doc_id="wiki",
                chunk_id=title,
            )
        )
    return format_hits(hits)


def web_search(
    query: str,
    *,
    searxng_url: str = "",
    serper_key: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    q = (query or "").strip()
    if not q:
        return EMPTY_RESULT
    if searxng_url:
        return _searxng(q, searxng_url, timeout)
    if serper_key:
        return _serper(q, serper_key, timeout)
    return _duckduckgo(q, timeout)


def _searxng(query: str, base_url: str, timeout: float) -> str:
    root = (base_url or "").rstrip("/")
    if not root or _is_blocked_url(root):
        return EMPTY_RESULT
    try:
        with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=True) as client:
            res = client.get(
                f"{root}/search",
                params={"q": query, "format": "json"},
            )
            res.raise_for_status()
            payload = res.json()
    except (httpx.HTTPError, ValueError):
        return EMPTY_RESULT
    results = payload.get("results") if isinstance(payload, dict) else None
    return _format_web(results if isinstance(results, list) else [])


def _serper(query: str, api_key: str, timeout: float) -> str:
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            res = client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
            )
            res.raise_for_status()
            payload = res.json()
    except (httpx.HTTPError, ValueError):
        return EMPTY_RESULT
    results = payload.get("organic") if isinstance(payload, dict) else None
    return _format_web(results if isinstance(results, list) else [])


def _format_web(results: list[Any]) -> str:
    hits: list[KnowledgeHit] = []
    for item in results[:5]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or title).strip()
        if not url or _is_blocked_url(url):
            continue
        hits.append(
            KnowledgeHit(
                text=snippet or title,
                source=url,
                score=1.0,
                doc_id="web",
                chunk_id=url,
            )
        )
    return format_hits(hits)


def _duckduckgo(query: str, timeout: float) -> str:
    if _is_blocked_url(DDG_HTML):
        return EMPTY_RESULT
    try:
        with httpx.Client(
            timeout=timeout, trust_env=False, follow_redirects=True
        ) as client:
            res = client.get(
                DDG_HTML,
                params={"q": query},
                headers={"User-Agent": _DDG_UA},
            )
    except httpx.HTTPError:
        return EMPTY_RESULT
    if res.status_code != 200:
        return EMPTY_RESULT
    return _format_ddg(_parse_ddg_html(res.text or ""))


def _format_ddg(items: list[tuple[str, str, str]]) -> str:
    lines: list[str] = []
    for title, url, snippet in items[:5]:
        dest = _unwrap_ddg_url(url)
        if dest and _is_blocked_url(dest):
            continue
        head = unescape(title).strip()
        body = unescape(snippet).strip() or head
        if not head:
            continue
        lines.append(f"[来源：网页搜索] {head} — {body}")
    return "\n".join(lines) if lines else EMPTY_RESULT


def _unwrap_ddg_url(href: str) -> str:
    raw = (href or "").strip()
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    uddg = parse_qs(parsed.query).get("uddg") or []
    if uddg:
        return unquote(uddg[0])
    return raw


def _parse_ddg_html(html: str) -> list[tuple[str, str, str]]:
    parser = _DDGParser()
    try:
        parser.feed(html or "")
        parser.close()
    except (ValueError, TypeError):
        return []
    return parser.items


class _DDGParser(HTMLParser):
    """顺序配对 result__a（标题）与 result__snippet（摘要）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._buf: list[str] = []
        self._href = ""
        self._pending_title = ""
        self._pending_href = ""
        self._open_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = " ".join(
            str(value or "") for key, value in attrs if key == "class"
        ).split()
        href = next((str(value or "") for key, value in attrs if key == "href"), "")
        if "result__a" in classes:
            self._flush_pending()
            self._in_title = True
            self._in_snippet = False
            self._href = href
            self._buf = []
            self._open_tag = tag
            return
        if "result__snippet" in classes:
            self._in_snippet = True
            self._in_title = False
            self._buf = []
            self._open_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == self._open_tag:
            self._pending_title = "".join(self._buf).strip()
            self._pending_href = self._href
            self._in_title = False
            self._buf = []
            return
        if self._in_snippet and tag == self._open_tag:
            snippet = "".join(self._buf).strip()
            if self._pending_title:
                self.items.append(
                    (self._pending_title, self._pending_href, snippet)
                )
                self._pending_title = ""
                self._pending_href = ""
            self._in_snippet = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_title or self._in_snippet:
            self._buf.append(data)

    def close(self) -> None:
        self._flush_pending()
        super().close()

    def _flush_pending(self) -> None:
        if self._pending_title:
            self.items.append((self._pending_title, self._pending_href, ""))
            self._pending_title = ""
            self._pending_href = ""
