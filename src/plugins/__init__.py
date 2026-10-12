from __future__ import annotations

from pocket_agent.plugins.agent_loop import AgentLoopPlugin
from pocket_agent.plugins.agents_plugin import AgentsPlugin
from pocket_agent.plugins.bridge_mcp import BridgeMcpPlugin
from pocket_agent.plugins.checkpoint_plugin import CheckpointPlugin
from pocket_agent.plugins.knowledge_plugin import KnowledgePlugin
from pocket_agent.plugins.llm_llamacpp import LlmLlamaCppPlugin
from pocket_agent.plugins.orchestrator_plugin import OrchestratorPlugin
from pocket_agent.plugins.prompts_plugin import PromptsPlugin
from pocket_agent.plugins.session_plugin import SessionPlugin
from pocket_agent.plugins.tools_core import ToolsCorePlugin
from pocket_agent.plugins.ui_cli import UiCliPlugin
from pocket_agent.plugins.ui_tui import UiTuiPlugin
from pocket_agent.plugins.ui_web import UiWebPlugin
from pocket_agent.plugins.workspace_plugin import WorkspacePlugin
from pocket_agent.plugins.writer_plugin import WriterPlugin
from pocket_agent.runtime import Plugin


def builtin_plugins() -> dict[str, Plugin]:
    """内置插件名 → 实例（供 pocket.yml 装配）。"""
    plugins: list[Plugin] = [
        PromptsPlugin(),
        LlmLlamaCppPlugin(),
        AgentsPlugin(),
        ToolsCorePlugin(),
        WorkspacePlugin(),
        KnowledgePlugin(),
        BridgeMcpPlugin(),
        CheckpointPlugin(),
        SessionPlugin(),
        WriterPlugin(),
        AgentLoopPlugin(),
        OrchestratorPlugin(),
        UiCliPlugin(),
        UiWebPlugin(),
        UiTuiPlugin(),
    ]
    return {p.name: p for p in plugins}
