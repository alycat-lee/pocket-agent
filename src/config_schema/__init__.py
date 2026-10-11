from __future__ import annotations

from dataclasses import fields
from typing import Any, Literal

from pocket_agent.config import AgentConfig

FieldType = Literal["string", "number", "integer", "boolean"]


def _item(
    key: str,
    *,
    type: FieldType,
    group: str,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
    enum: list[Any] | None = None,
    secret: bool = False,
    deprecated: bool = False,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "key": key,
        "type": type,
        "group": group,
        "description": description,
    }
    if minimum is not None:
        item["min"] = minimum
    if maximum is not None:
        item["max"] = maximum
    if enum is not None:
        item["enum"] = enum
    if secret:
        item["secret"] = True
    if deprecated:
        item["deprecated"] = True
    return item


# 声明式配置元数据——设置页唯一数据源；新增 AgentConfig 字段必须在此登记
CONFIG_SCHEMA: list[dict[str, Any]] = [
    # 模型（Phase 20 后由 providers 列表管理，扁平字段标记 deprecated）
    _item(
        "base_url",
        type="string",
        group="模型",
        description="LLM OpenAI 兼容接口地址（已迁移至 providers）",
        deprecated=True,
    ),
    _item(
        "model",
        type="string",
        group="模型",
        description="模型名称（已迁移至 providers）",
        deprecated=True,
    ),
    _item(
        "api_key",
        type="string",
        group="模型",
        description="云端 API Key（已迁移至 providers，只写不读）",
        secret=True,
        deprecated=True,
    ),
    _item(
        "temperature",
        type="number",
        group="模型",
        description="采样温度（已迁移至 providers sampling）",
        minimum=0.0,
        maximum=2.0,
        deprecated=True,
    ),
    _item(
        "repeat_penalty",
        type="number",
        group="模型",
        description="重复惩罚（已迁移至 providers sampling）",
        minimum=1.0,
        maximum=2.0,
        deprecated=True,
    ),
    _item(
        "top_p",
        type="number",
        group="模型",
        description="核采样 top_p（已迁移至 providers sampling）",
        minimum=0.0,
        maximum=1.0,
        deprecated=True,
    ),
    _item(
        "max_tokens",
        type="integer",
        group="模型",
        description="单次生成最大输出 token（已迁移至 providers sampling）",
        minimum=1,
        maximum=128000,
        deprecated=True,
    ),
    _item(
        "timeout",
        type="number",
        group="模型",
        description="HTTP 请求超时（秒，已迁移至 providers）",
        minimum=1.0,
        maximum=600.0,
        deprecated=True,
    ),
    _item(
        "reasoning_enabled",
        type="boolean",
        group="模型",
        description="是否保留思维链（仅轨迹检查器可见，默认关闭）",
    ),
    # 上下文
    _item(
        "prompt_char_budget",
        type="integer",
        group="上下文预算",
        description="提示词字符预算",
        minimum=500,
        maximum=100000,
    ),
    _item(
        "per_result_limit",
        type="integer",
        group="上下文预算",
        description="单条工具结果压缩上限（字符）",
        minimum=50,
        maximum=10000,
    ),
    _item(
        "summarize_input_limit",
        type="integer",
        group="上下文预算",
        description="摘要器输入上限（字符）",
        minimum=100,
        maximum=50000,
    ),
    _item(
        "use_llm_summarize",
        type="boolean",
        group="上下文预算",
        description="是否用 LLM 做结果摘要（默认规则截断）",
    ),
    # 护栏
    _item(
        "max_steps",
        type="integer",
        group="护栏",
        description="单轮最大工具步数",
        minimum=1,
        maximum=100,
    ),
    _item(
        "max_consecutive_intercepts",
        type="integer",
        group="护栏",
        description="连续拦截上限后停止",
        minimum=1,
        maximum=20,
    ),
    # 会话
    _item(
        "keep_full_turns",
        type="integer",
        group="会话",
        description="保留完整记录的轮数",
        minimum=1,
        maximum=100,
    ),
    _item(
        "session_summary_limit",
        type="integer",
        group="会话",
        description="滚动摘要字符上限",
        minimum=50,
        maximum=10000,
    ),
    # 长文
    _item(
        "section_char_target",
        type="integer",
        group="长文写作",
        description="每段目标字数",
        minimum=100,
        maximum=10000,
    ),
    _item(
        "seam_tail_chars",
        type="integer",
        group="长文写作",
        description="接缝上下文长度",
        minimum=0,
        maximum=5000,
    ),
    _item(
        "overlap_max",
        type="integer",
        group="长文写作",
        description="拼接去重最大重叠窗口",
        minimum=0,
        maximum=2000,
    ),
    # 工具
    _item(
        "tool_recall_threshold",
        type="integer",
        group="工具",
        description="超过此工具数启用关键词召回",
        minimum=1,
        maximum=1000,
    ),
    _item(
        "tool_recall_top_n",
        type="integer",
        group="工具",
        description="召回返回的候选工具数",
        minimum=1,
        maximum=50,
    ),
    # 路由（Phase 20 后由 providers routing 管理，标记 deprecated）
    _item(
        "route_cloud",
        type="boolean",
        group="模型路由",
        description="复杂生成走云端（已迁移至 providers routing）",
        deprecated=True,
    ),
    # 规划与验收
    _item(
        "planner_enabled",
        type="boolean",
        group="规划验收",
        description="是否启用 Planner 任务卡（关闭则退回三分类）",
    ),
    _item(
        "verifier_enabled",
        type="boolean",
        group="规划验收",
        description="交付前是否做纯代码验收",
    ),
    _item(
        "answer_max_retries",
        type="integer",
        group="规划验收",
        description="验收失败后的重试次数",
        minimum=0,
        maximum=5,
    ),
    _item(
        "agents_root",
        type="string",
        group="Agent",
        description="Agent 根目录路径。修改后只影响新创建的 agent，不会搬迁已有目录。",
    ),
]


def get_config_schema() -> list[dict[str, Any]]:
    """返回配置 schema 副本（设置页数据源）；deprecated 项不返回。"""
    return [dict(item) for item in CONFIG_SCHEMA if not item.get("deprecated")]


def schema_keys() -> set[str]:
    return {str(item["key"]) for item in CONFIG_SCHEMA}


def assert_schema_covers_agent_config() -> None:
    """开发期断言：AgentConfig 每个字段都在 schema 中登记。"""
    config_keys = {f.name for f in fields(AgentConfig)}
    missing = config_keys - schema_keys()
    extra = schema_keys() - config_keys
    if missing or extra:
        raise AssertionError(
            f"config schema mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
