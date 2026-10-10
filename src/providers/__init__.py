"""多 provider 注册表：按 id 解析 client，routing 决定原子任务走哪个 provider。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import (
    LlamaCppClient,
    LLMClientProtocol,
)

ProviderType = Literal["llamacpp", "openai_compatible", "deepseek"]


@dataclass
class ProviderConfig:
    id: str
    name: str
    type: ProviderType
    base_url: str
    api_key: str = ""
    model: str = ""
    sampling: dict[str, Any] | None = None  # 可空覆盖 AgentConfig 采样参数


@dataclass
class RoutingConfig:
    choose_fill: str  # provider id（choose_tool / fill_args）
    generate: str  # provider id（final_answer / summarize / rewrite）


@dataclass
class ProvidersConfig:
    providers: list[ProviderConfig] = field(default_factory=list)
    default_provider_id: str = ""
    routing: RoutingConfig = field(
    default_factory=lambda: RoutingConfig(choose_fill="", generate="")
    )


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def migrate_providers(
    settings: dict[str, Any],
    base_config: AgentConfig,
) -> dict[str, Any]:
    """旧扁平模型字段 → providers[0]；已迁移则跳过。"""
    if "providers" in settings and isinstance(settings["providers"], list):
        return settings
    pid = "local"
    providers = [
        {
            "id": pid,
            "name": "本地模型",
            "type": "llamacpp",
            "base_url": base_config.base_url,
            "api_key": base_config.api_key,
            "model": base_config.model,
        }
    ]
    settings = dict(settings)
    settings["providers"] = providers
    settings["default_provider_id"] = pid
    settings["routing"] = {"choose_fill": pid, "generate": pid}
    return settings


def load_providers_config(
    settings: dict[str, Any],
    base_config: AgentConfig,
) -> ProvidersConfig:
    """从 settings.json 构造 ProvidersConfig；不存在时自动迁移。"""
    migrated = migrate_providers(settings, base_config)
    raw_list = migrated.get("providers") or []
    providers: list[ProviderConfig] = []
    for item in raw_list:
        if not isinstance(item, dict) or "id" not in item:
            continue
        providers.append(
            ProviderConfig(
                id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                type=item.get("type", "llamacpp"),  # type: ignore[arg-type]
                base_url=str(item.get("base_url") or base_config.base_url),
                api_key=str(item.get("api_key") or ""),
                model=str(item.get("model") or base_config.model),
                sampling=item.get("sampling") if isinstance(item.get("sampling"), dict) else None,
            )
        )
    default_id = str(migrated.get("default_provider_id") or "")
    routing_raw = migrated.get("routing") or {}
    routing = RoutingConfig(
        choose_fill=str(routing_raw.get("choose_fill") or default_id or ""),
        generate=str(routing_raw.get("generate") or default_id or ""),
    )
    if not routing.choose_fill and providers:
        routing.choose_fill = providers[0].id
    if not routing.generate and providers:
        routing.generate = providers[0].id
    return ProvidersConfig(
        providers=providers,
        default_provider_id=default_id or (providers[0].id if providers else ""),
        routing=routing,
    )


def providers_to_dict(cfg: ProvidersConfig) -> dict[str, Any]:
    return {
        "providers": [
            {
                "id": p.id,
                "name": p.name,
                "type": p.type,
                "base_url": p.base_url,
                "api_key": p.api_key,
                "model": p.model,
                "sampling": p.sampling,
            }
            for p in cfg.providers
        ],
        "default_provider_id": cfg.default_provider_id,
        "routing": {
            "choose_fill": cfg.routing.choose_fill,
            "generate": cfg.routing.generate,
        },
    }


def redact_providers(cfg: ProvidersConfig) -> dict[str, Any]:
    """api_key 只写不读：返回时脱敏。"""
    data = providers_to_dict(cfg)
    for p in data["providers"]:
        key = p.get("api_key", "")
        if isinstance(key, str) and key:
            p["api_key"] = key[:3] + "***" if len(key) > 3 else "***"
    return data


class ProviderRegistry:
    """按 routing 规则把原子任务分流到不同 provider 的 LLM 服务。

    - generate → routing.generate（final_answer / summarize / rewrite）
    - generate_int / generate_json → routing.choose_fill（choose_tool / fill_args）
    实现 LLMClientProtocol，对外是唯一的 llm 服务。
    """

    def __init__(
        self,
        config: ProvidersConfig,
        base: AgentConfig,
    ) -> None:
        self._config = config
        self._base = base
        self._clients: dict[str, LLMClientProtocol] = {}

    def _build_client(self, pc: ProviderConfig) -> LLMClientProtocol:
        ac = replace(
            self._base,
            base_url=pc.base_url,
            model=pc.model or self._base.model,
            api_key=pc.api_key,
        )
        if pc.sampling:
            for k, v in pc.sampling.items():
                if k in AgentConfig.known_keys() and v is not None:
                    ac = replace(ac, **{k: v})
        if pc.type == "llamacpp":
            return LlamaCppClient(ac)
        # openai_compatible / deepseek：延迟导入避免 providers → plugins 循环依赖
        from pocket_agent.plugins.llm_deepseek import DeepSeekClient

        return DeepSeekClient(
            api_key=pc.api_key,
            base_url=pc.base_url,
            model=pc.model or "local-model",
            temperature=ac.temperature,
            top_p=ac.top_p,
            max_tokens=ac.max_tokens,
            timeout=ac.timeout,
            trust_env=pc.type != "llamacpp",
        )

    def _get_client(self, provider_id: str) -> LLMClientProtocol:
        if provider_id not in self._clients:
            pc = next(
                (p for p in self._config.providers if p.id == provider_id), None
            )
            if pc is None:
                raise RuntimeError(f"provider not found: {provider_id}")
            self._clients[provider_id] = self._build_client(pc)
        return self._clients[provider_id]

    def generate(self, prompt: str) -> str:
        return self._get_client(self._config.routing.generate).generate(prompt)

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        return self._get_client(self._config.routing.choose_fill).generate_int(
            prompt, minimum, maximum
        )

    def generate_json(self, prompt: str, schema: dict) -> dict:
        return self._get_client(self._config.routing.choose_fill).generate_json(
            prompt, schema
        )
