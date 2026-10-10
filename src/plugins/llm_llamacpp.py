from __future__ import annotations

from pocket_agent.llm_client import LlamaCppClient
from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context


class LlmLlamaCppPlugin:
    """本地 llama.cpp OpenAI 兼容客户端 → llm / llm_local。"""

    name = "llm-llamacpp"
    requires: list[str]
    provides: list[str]

    def __init__(self) -> None:
        self.requires = []
        self.provides = ["llm", "llm_local"]

    def prepare(self, config: dict) -> None:
        """local_only=true 时只提供 llm_local，供云端路由组合。"""
        if bool(config.get("local_only")):
            self.provides = ["llm_local"]
        else:
            self.provides = ["llm", "llm_local"]

    def apply(self, ctx: Context, config: dict) -> None:
        ac = agent_config_from(config)
        client = LlamaCppClient(ac)
        ctx.provide("llm_local", client)
        if bool(config.get("local_only")):
            self.provides = ["llm_local"]
            return
        if ctx.has("llm"):
            self.provides = ["llm_local"]
            return
        ctx.provide("llm", client)
        self.provides = ["llm", "llm_local"]
