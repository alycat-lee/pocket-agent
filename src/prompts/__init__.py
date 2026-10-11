# 运行时提示词模板：措辞必须与 pocket-agent-docs/04-运行时提示词模板.md 逐字一致。

SYSTEM_BASE: str = """你是一个任务执行助手，通过调用工具逐步完成任务。

规则：
1. 每一步只做一件事
2. 不要重复调用同一个工具和相同的参数，已有结果直接使用
3. 严格按照要求的格式输出，不要输出任何解释"""

CHOOSE_TOOL: str = """任务：{task}

当前进度：{state}

已执行的历史（最近的在前）：
{history_text}

可用工具：
{options}

下一步应该使用哪个工具？如果已有信息足以回答任务，输出 -1。
只输出一个数字。例如：0"""

FILL_ARGS: str = """任务：{task}

已知信息：{state}

请为工具 {tool_name} 生成调用参数。
参数要求：{schema_json}

只输出 JSON 对象，不要输出任何其他内容。"""

SUMMARIZE: str = """将以下内容压缩为不超过 {limit} 字的摘要，保留所有关键数据（数字、名称、结论）：

{text}

只输出摘要本身。"""

FINAL_ANSWER: str = """任务：{task}

以下是通过工具调用收集到的信息：
{history_text}

基于以上信息回答任务。如果信息不足以完整回答，说明缺失的部分。
直接输出回答，不要复述过程。"""

REWRITE_QUERY: str = """根据对话历史，把用户的最新输入改写成一个完整、独立的问题。
不要回答，只输出改写后的问题。

对话历史：
{history_text}

最新输入：{user_input}

改写后的问题："""

ROLLING_SUMMARY: str = """把这段对话历史压缩为不超过 {limit} 字的摘要，
保留所有关键事实（数字、名称、用户提供的个人信息）：

{turns_text}

只输出摘要本身。"""

OUTLINE: str = """为主题「{topic}」生成一篇约 {total_chars} 字文章的大纲。
每段给出标题和 2~3 个要点，每段约 {section_chars} 字。
参考材料：{material}

只输出 JSON。"""

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

SECTION_WRITE: str = """你正在逐段撰写文章「{topic}」。
全文大纲：{outline_text}
必须保持一致的事实：{consistency_text}

请撰写第 {index} 段「{section_title}」，要点：{points}
要求：约 {target_chars} 字；承接上文，不要重复前文已写内容。

上文结尾：{seam_text}

只输出本段正文。"""

SESSION_TITLE: str = """为以下对话生成一个不超过 12 个字的会话标题，只输出标题本身。
用户：{user_input}"""

INTENT_ROUTE: str = """判断用户意图类型。0=直接问答，不需要工具；
1=需要调用工具完成任务；2=需要撰写长文。
用户输入：{user_input} 只输出数字。"""

PLANNER: str = """分析用户任务，输出任务卡 JSON。

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
