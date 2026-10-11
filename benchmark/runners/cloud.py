"""跑道 C：DeepSeek API（统一 OpenAICompatibleClient + deepseek 预设）。"""

from __future__ import annotations

import os
from typing import Any

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LLMConnectionError, LLMFormatError, OpenAICompatibleClient
from pocket_agent.providers import PROVIDER_PRESETS

from benchmark.runners import TrackResult
from benchmark.schema import task_turns

API_KEY_ENV = "DEEPSEEK_API_KEY"
SKIP_REASON = f"{API_KEY_ENV} 缺失"


def _deepseek_client(*, max_tokens: int, timeout: float) -> OpenAICompatibleClient | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        return None
    preset = PROVIDER_PRESETS["deepseek"]
    ac = AgentConfig(
        base_url=str(preset["base_url"]),
        model="deepseek-chat",
        api_key=key,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return OpenAICompatibleClient(
        ac,
        trust_env=bool(preset["trust_env"]),
        thinking=preset["thinking"],
        api_key=key,
    )


def cloud_available() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def make_judge_client(*, max_tokens: int = 512, timeout: float = 60.0) -> OpenAICompatibleClient | None:
    return _deepseek_client(max_tokens=max_tokens, timeout=timeout)


def _quota_or_auth(exc: Exception) -> bool:
    text = str(exc)
    return any(tok in text for tok in ("401", "402", "403", "Insufficient Balance"))


class CloudRunner:
    name = "cloud"

    def __init__(self, *, max_tokens: int = 1024, timeout: float = 60.0) -> None:
        self._client = _deepseek_client(max_tokens=max_tokens, timeout=timeout)
        self._dead_reason = "" if self._client is not None else SKIP_REASON

    def run_task(self, task: dict[str, Any]) -> TrackResult:
        tid = str(task["id"])
        dim = str(task["dimension"])
        if self._client is None or self._dead_reason:
            return TrackResult(
                track=self.name,
                task_id=tid,
                dimension=dim,
                answer="",
                skipped=True,
                skip_reason=self._dead_reason or SKIP_REASON,
            )
        transcript = ""
        answers: list[str] = []
        try:
            for user in task_turns(task):
                prompt = _history_prompt(transcript, user)
                text = self._client.generate(prompt)
                answers.append(text)
                transcript += f"User: {user}\nAssistant: {text}\n"
        except (LLMConnectionError, LLMFormatError) as exc:
            if _quota_or_auth(exc):
                self._dead_reason = "DeepSeek 额度/鉴权失败"
                return TrackResult(
                    track=self.name,
                    task_id=tid,
                    dimension=dim,
                    answer=answers[-1] if answers else "",
                    turn_answers=answers,
                    skipped=True,
                    skip_reason=self._dead_reason,
                )
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


def _history_prompt(transcript: str, user: str) -> str:
    if not transcript.strip():
        return user
    return (
        "以下是到目前为止的对话，请只回答用户的最新一句。\n\n"
        f"{transcript}\nUser: {user}\nAssistant:"
    )
