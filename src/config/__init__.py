from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS_PATH = Path("settings.json")


@dataclass
class AgentConfig:
    # LLM 服务
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"  # llama.cpp server 不校验模型名
    api_key: str = ""  # 云端密钥；GET 配置时脱敏
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

    # 模型路由（Phase 14/17）
    route_cloud: bool = False  # True：复杂生成走云端

    @classmethod
    def known_keys(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> AgentConfig:
        """仅接受已知字段，忽略未知键。"""
        raw = dict(data or {})
        known = cls.known_keys()
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def from_layers(
        cls,
        base: AgentConfig | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AgentConfig:
        """三层合并：pocket 默认 < settings.json < 请求 overrides（后者优先）。"""
        merged: dict[str, Any] = asdict(cls())
        if base is not None:
            if isinstance(base, AgentConfig):
                merged.update(asdict(base))
            else:
                merged.update({k: v for k, v in base.items() if k in cls.known_keys()})
        if settings:
            merged.update({k: v for k, v in settings.items() if k in cls.known_keys()})
        if overrides:
            merged.update({k: v for k, v in overrides.items() if k in cls.known_keys()})
        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_base_from_pocket(pocket: dict[str, Any] | None) -> dict[str, Any]:
    """从 pocket.yml 插件 config 中收集 AgentConfig 字段作为默认层。"""
    known = AgentConfig.known_keys()
    merged: dict[str, Any] = {}
    if not pocket:
        return merged
    plugins = pocket.get("plugins") or []
    if not isinstance(plugins, list):
        return merged
    # 靠后的插件 config 覆盖靠前的同名字段
    for entry in plugins:
        if not isinstance(entry, dict) or entry.get("disabled"):
            continue
        cfg = entry.get("config") or {}
        if not isinstance(cfg, dict):
            continue
        for key, value in cfg.items():
            if key in known:
                merged[key] = value
    return merged


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """读取用户设置层；损坏时告警并回退空 dict（即 pocket 默认）。"""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_file():
        return {}
    try:
        raw = settings_path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        msg = f"settings.json corrupt ({settings_path}): {exc}; falling back to pocket.yml defaults"
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return {}
    if not isinstance(data, dict):
        msg = f"settings.json root must be a mapping ({settings_path}); falling back to defaults"
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return {}
    known = AgentConfig.known_keys()
    return {k: v for k, v in data.items() if k in known}


def save_settings(
    data: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    """原子写入 settings.json（写临时文件再 rename）。"""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    known = AgentConfig.known_keys()
    clean = {k: v for k, v in data.items() if k in known}
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(settings_path)


def redact_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """GET /api/config 脱敏：api_key 只写不读明文。"""
    out = dict(data)
    key = out.get("api_key")
    if isinstance(key, str) and key:
        if len(key) <= 3:
            out["api_key"] = "***"
        else:
            out["api_key"] = key[:3] + "***"
    elif "api_key" in out:
        out["api_key"] = ""
    return out
