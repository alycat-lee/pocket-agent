"""外网检索：Wikipedia + SearXNG/Serper，结果必须带来源标记。"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from pocket_agent.knowledge import EMPTY_RESULT, KnowledgeHit, format_hits

WIKI_API = "https://zh.wikipedia.org/w/api.php"
DEFAULT_TIMEOUT = 8.0


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
    return EMPTY_RESULT


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
