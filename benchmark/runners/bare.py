"""跑道 A：直接对 llama-server 多轮自由对话，无框架机制。"""

from __future__ import annotations

from typing import Any

import httpx

from benchmark.runners import TrackResult
from benchmark.schema import task_turns


class BareRunner:
    name = "bare"

    def __init__(self, base_url: str, *, timeout: float = 120.0, max_tokens: int = 512) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_tokens = max_tokens

    def run_task(self, task: dict[str, Any]) -> TrackResult:
        tid = str(task["id"])
        dim = str(task["dimension"])
        messages: list[dict[str, str]] = []
        answers: list[str] = []
        try:
            for user in task_turns(task):
                messages.append({"role": "user", "content": user})
                text = self._chat(messages)
                answers.append(text)
                messages.append({"role": "assistant", "content": text})
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            return TrackResult(
                track=self.name,
                task_id=tid,
                dimension=dim,
                answer=answers[-1] if answers else "",
                turn_answers=answers,
                error=str(exc),
            )
        return TrackResult(
            track=self.name,
            task_id=tid,
            dimension=dim,
            answer=answers[-1] if answers else "",
            turn_answers=answers,
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self._max_tokens,
        }
        with httpx.Client(timeout=self._timeout, trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        raw = data["choices"][0]["message"].get("content")
        return raw if isinstance(raw, str) else ""
