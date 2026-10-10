from __future__ import annotations

from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.checkpoint_plugin import CheckpointPlugin
from pocket_agent.plugins.llm_deepseek import LlmDeepSeekPlugin
from pocket_agent.plugins.llm_llamacpp import LlmLlamaCppPlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.ui_cli import UiCliPlugin
from pocket_agent.plugins.writer_plugin import WriterPlugin
from pocket_agent.runtime import Plugin


def builtin_plugins() -> dict[str, Plugin]:
    """内置插件名 → 实例（供 pocket.yml 装配）。"""
    plugins: list[Plugin] = [
        LlmLlamaCppPlugin(),
        LlmDeepSeekPlugin(),
        ToolsCorePlugin(),
        CheckpointPlugin(),
        SessionPlugin(),
        WriterPlugin(),
        AgentLoopPlugin(),
        UiCliPlugin(),
    ]
    return {p.name: p for p in plugins}
