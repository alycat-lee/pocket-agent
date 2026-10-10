from dataclasses import dataclass


@dataclass
class AgentConfig:
    # LLM 服务
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"  # llama.cpp server 不校验模型名
    timeout: float = 60.0

    # 采样参数
    temperature: float = 0.2
    repeat_penalty: float = 1.1
    top_p: float = 0.9
    max_tokens: int = 512  # 单次生成的最大输出 token

    # 上下文预算（字符数）
    prompt_char_budget: int = 5200  # ≈ 3500 tokens（× 1.5 换算）
    per_result_limit: int = 300  # 单条工具结果压缩后上限
    summarize_input_limit: int = 2000  # 喂给摘要器的输入上限
    use_llm_summarize: bool = False  # 默认规则截断，不开 LLM 摘要

    # 护栏
    max_steps: int = 10
    max_consecutive_intercepts: int = 2

    # 工具检索
    tool_recall_threshold: int = 8  # 注册工具数超过此值启用召回
    tool_recall_top_n: int = 5

    # 会话
    keep_full_turns: int = 6  # 保留完整记录的轮数
    session_summary_limit: int = 500  # 滚动摘要字符上限

    # 长文本（Phase 12）
    section_char_target: int = 500  # 每段目标字数
    seam_tail_chars: int = 250  # 接缝上下文长度
    overlap_max: int = 80  # 拼接去重的最大重叠窗口
