# 运行时提示词模板：cloud 变体与 pocket-agent-docs/04 逐字一致。
# local 变体是文档 18 要求的超短指令（≤150 字），由 prompt_registry 按 provider 选用。

from __future__ import annotations

from typing import Any

from pocket_agent.prompt_registry import resolve_prompt

_CLOUD_SYSTEM_BASE: str = """你是一个任务执行助手，通过调用工具逐步完成任务。

规则：
1. 每一步只做一件事
2. 不要重复调用同一个工具和相同的参数，已有结果直接使用
3. 严格按照要求的格式输出，不要输出任何解释"""

_CLOUD_CHOOSE_TOOL: str = """任务：{task}

当前进度：{state}

已执行的历史（最近的在前）：
{history_text}

可用工具：
{options}

下一步应该使用哪个工具？如果已有信息足以回答任务，输出 -1。
只输出一个数字。例如：0"""

_CLOUD_FILL_ARGS: str = """任务：{task}

已知信息：{state}

请为工具 {tool_name} 生成调用参数。
参数要求：{schema_json}

只输出 JSON 对象，不要输出任何其他内容。"""

_CLOUD_SUMMARIZE: str = """将以下内容压缩为不超过 {limit} 字的摘要，保留所有关键数据（数字、名称、结论）：

{text}

只输出摘要本身。"""

_CLOUD_FINAL_ANSWER: str = """任务：{task}

以下是通过工具调用收集到的信息：
{history_text}

基于以上信息回答任务。如果信息不足以完整回答，说明缺失的部分。
直接输出回答，不要复述过程。"""

_CLOUD_REWRITE_QUERY: str = """根据对话历史，把用户的最新输入改写成一个完整、独立的问题。
不要回答，只输出改写后的问题。

对话历史：
{history_text}

最新输入：{user_input}

改写后的问题："""

_CLOUD_ROLLING_SUMMARY: str = """把这段对话历史压缩为不超过 {limit} 字的摘要，
保留所有关键事实（数字、名称、用户提供的个人信息）：

{turns_text}

只输出摘要本身。"""

_CLOUD_OUTLINE: str = """为主题「{topic}」生成一篇约 {total_chars} 字文章的大纲。
每段给出标题和 2~3 个要点，每段约 {section_chars} 字。
参考材料：{material}

只输出 JSON。"""

_CLOUD_SECTION_WRITE: str = """你正在逐段撰写文章「{topic}」。
全文大纲：{outline_text}
必须保持一致的事实：{consistency_text}

请撰写第 {index} 段「{section_title}」，要点：{points}
要求：约 {target_chars} 字；承接上文，不要重复前文已写内容。

上文结尾：{seam_text}

只输出本段正文。"""

_CLOUD_SESSION_TITLE: str = """为以下对话生成一个不超过 12 个字的会话标题，只输出标题本身。
用户：{user_input}"""

_CLOUD_INTENT_ROUTE: str = """判断用户意图类型。0=直接问答，不需要工具；
1=需要调用工具完成任务；2=需要撰写长文。
用户输入：{user_input} 只输出数字。"""

_CLOUD_PLANNER: str = """分析用户任务，输出任务卡 JSON。

要求：
1. type：直接问答=qa，需要工具=agent，创作长文=writer
2. goal：一句话概括目标
3. constraints：提取所有硬性要求（字数、格式、语言等），
   字数统一换算为数字（"2万字" → "total_chars": 20000）
4. plan：3~7 个执行步骤，每步一句话
5. needs_history：任务是否依赖之前的对话内容

会话摘要：{rolling_summary}
用户任务：{user_input}

只输出 JSON。"""

_LOCAL_SYSTEM_BASE: str = (
    "你是任务助手，通过工具逐步完成任务。每步只做一件事。"
    "不要重复同一工具和参数。只按要求格式输出。"
)

_LOCAL_CHOOSE_TOOL: str = """任务：{task}
进度：{state}
历史：{history_text}
工具：{options}
选编号；够了输出 -1。只输出数字。例如：0"""

_LOCAL_FILL_ARGS: str = """任务：{task}
已知：{state}
为 {tool_name} 填参：{schema_json}
只输出 JSON 对象。"""

_LOCAL_SUMMARIZE: str = """压缩为不超过 {limit} 字，保留数字、名称、结论：
{text}
只输出摘要本身。"""

_LOCAL_FINAL_ANSWER: str = """任务：{task}
信息：{history_text}
据此回答；不足则说明缺什么。直接输出，不要复述过程。"""

_LOCAL_REWRITE_QUERY: str = """根据历史把最新输入改成完整独立问题。不要回答。
历史：{history_text}
输入：{user_input}"""

_LOCAL_ROLLING_SUMMARY: str = """压缩对话为不超过 {limit} 字，保留数字、名称、个人信息：
{turns_text}
只输出摘要本身。"""

_LOCAL_OUTLINE: str = (
    "为主题「{topic}」写约 {total_chars} 字大纲。"
    "每段标题+2~3要点，每段约 {section_chars} 字。材料：{material}\n只输出 JSON。"
)

_LOCAL_SECTION_WRITE: str = """写「{topic}」第 {index} 段「{section_title}」，约 {target_chars} 字。
大纲：{outline_text}
事实：{consistency_text}
要点：{points}
上文：{seam_text}
只输出本段正文。"""

_LOCAL_SESSION_TITLE: str = """为对话生成不超过 12 字标题，只输出标题。
用户：{user_input}"""

_LOCAL_INTENT_ROUTE: str = """判断意图。0=直接问答；1=要工具；2=写长文。
输入：{user_input} 只输出数字。"""

_LOCAL_PLANNER: str = """输出任务卡 JSON。type=qa/agent/writer；goal；constraints（字数换数字）；plan 3~7步；needs_history。
摘要：{rolling_summary}
任务：{user_input}
只输出 JSON。"""

_CLOUD_SYNTHESIZE: str = """任务：{task}

以下是各子任务摘要（已压缩，不含全文）：
{summaries}

请合并为完整回答。只输出回答本身。"""

_LOCAL_SYNTHESIZE: str = """任务：{task}
摘要：{summaries}
合并为完整回答。只输出回答本身。"""

_BUILTIN_VERSION = "1.0.0"


class PromptRef(str):
    """str 兼容包装：.format / f-string 走当前 registry 变体，无 registry 时用 cloud 原文。"""

    _prompt_id: str
    _fallback: str

    def __new__(cls, prompt_id: str, fallback: str | None = None) -> str:
        # 切片 / 单参构造必须退回普通 str，避免 system[:n] 因缺 fallback 崩溃。
        if fallback is None:
            return str.__new__(str, prompt_id)
        obj = str.__new__(cls, fallback)
        obj._prompt_id = prompt_id
        obj._fallback = fallback
        return obj

    def _resolved(self) -> str:
        return resolve_prompt(self._prompt_id, fallback=self._fallback)

    def format(self, *args: object, **kwargs: object) -> str:
        return self._resolved().format(*args, **kwargs)

    def __str__(self) -> str:
        return self._resolved()

    def __format__(self, format_spec: str) -> str:
        return format(self._resolved(), format_spec)

    def __len__(self) -> int:
        return len(self._resolved())

    def __getitem__(self, key: int | slice) -> str:
        return self._resolved()[key]

    def __add__(self, other: object) -> str:
        if isinstance(other, str):
            return self._resolved() + other
        return NotImplemented

    def __radd__(self, other: object) -> str:
        if isinstance(other, str):
            return other + self._resolved()
        return NotImplemented


SYSTEM_BASE = PromptRef("system_base", _CLOUD_SYSTEM_BASE)
CHOOSE_TOOL = PromptRef("choose_tool", _CLOUD_CHOOSE_TOOL)
FILL_ARGS = PromptRef("fill_args", _CLOUD_FILL_ARGS)
SUMMARIZE = PromptRef("summarize", _CLOUD_SUMMARIZE)
FINAL_ANSWER = PromptRef("final_answer", _CLOUD_FINAL_ANSWER)
REWRITE_QUERY = PromptRef("rewrite_query", _CLOUD_REWRITE_QUERY)
ROLLING_SUMMARY = PromptRef("rolling_summary", _CLOUD_ROLLING_SUMMARY)
OUTLINE = PromptRef("outline", _CLOUD_OUTLINE)
SECTION_WRITE = PromptRef("section_write", _CLOUD_SECTION_WRITE)
SESSION_TITLE = PromptRef("session_title", _CLOUD_SESSION_TITLE)
INTENT_ROUTE = PromptRef("intent_route", _CLOUD_INTENT_ROUTE)
PLANNER = PromptRef("planner", _CLOUD_PLANNER)
SYNTHESIZE = PromptRef("synthesize", _CLOUD_SYNTHESIZE)

OUTLINE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "points"],
            },
        }
    },
    "required": ["sections"],
}

_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("system_base", _CLOUD_SYSTEM_BASE, _LOCAL_SYSTEM_BASE),
    ("choose_tool", _CLOUD_CHOOSE_TOOL, _LOCAL_CHOOSE_TOOL),
    ("fill_args", _CLOUD_FILL_ARGS, _LOCAL_FILL_ARGS),
    ("summarize", _CLOUD_SUMMARIZE, _LOCAL_SUMMARIZE),
    ("final_answer", _CLOUD_FINAL_ANSWER, _LOCAL_FINAL_ANSWER),
    ("rewrite_query", _CLOUD_REWRITE_QUERY, _LOCAL_REWRITE_QUERY),
    ("rolling_summary", _CLOUD_ROLLING_SUMMARY, _LOCAL_ROLLING_SUMMARY),
    ("outline", _CLOUD_OUTLINE, _LOCAL_OUTLINE),
    ("section_write", _CLOUD_SECTION_WRITE, _LOCAL_SECTION_WRITE),
    ("session_title", _CLOUD_SESSION_TITLE, _LOCAL_SESSION_TITLE),
    ("intent_route", _CLOUD_INTENT_ROUTE, _LOCAL_INTENT_ROUTE),
    ("planner", _CLOUD_PLANNER, _LOCAL_PLANNER),
    ("synthesize", _CLOUD_SYNTHESIZE, _LOCAL_SYNTHESIZE),
)


def builtin_entries() -> list[dict[str, Any]]:
    """内置原子任务提示词（cloud=文档 04 原文，local=超短指令）。"""
    return [
        {
            "id": prompt_id,
            "template": cloud,
            "variants": {"cloud": cloud, "local": local},
            "version": _BUILTIN_VERSION,
        }
        for prompt_id, cloud, local in _ENTRIES
    ]
