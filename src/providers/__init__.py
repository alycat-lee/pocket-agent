"""多 provider 注册表：按 id 解析 client，routing 决定原子任务走哪个 provider。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import (
    LLMClientProtocol,
    OpenAICompatibleClient,
    ThinkingMode,
)

ProviderType = Literal["llamacpp", "openai_compatible", "deepseek"]

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "llamacpp": {
        "base_url": "http://localhost:8000/v1",
        "requires_key": False,
        "trust_env": False,
        "thinking": "off",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "requires_key": True,
        "trust_env": True,
        "thinking": "strip",
    },
    "openai_compatible": {
        "requires_key": True,
        "trust_env": True,
        "thinking": "strip",
    },
}

PRESET_LABELS: dict[str, str] = {
    "llamacpp": "llama.cpp（本地）",
    "deepseek": "DeepSeek",
    "openai_compatible": "OpenAI 兼容",
}


def preset_catalog() -> list[dict[str, Any]]:
    """设置页下拉：预设 id + 中文标签 + 默认可改字段。"""
    catalog: list[dict[str, Any]] = []
    for key, preset in PROVIDER_PRESETS.items():
        item: dict[str, Any] = {"id": key, "label": PRESET_LABELS[key]}
        item.update(preset)
        catalog.append(item)
    return catalog


def apply_preset(name: str) -> dict[str, Any]:
    if name in PROVIDER_PRESETS:
        return dict(PROVIDER_PRESETS[name])
    return dict(PROVIDER_PRESETS["openai_compatible"])


@dataclass
class ProviderConfig:
    id: str
    name: str
    type: ProviderType
    base_url: str
    api_key: str = ""
    model: str = ""
    sampling: dict[str, Any] | None = None
    thinking: ThinkingMode | None = None
    trust_env: bool | None = None


@dataclass
class RoutingConfig:
    choose_fill: str
    generate: str


@dataclass
class ProvidersConfig:
    providers: list[ProviderConfig] = field(default_factory=list)
    default_provider_id: str = ""
    routing: RoutingConfig = field(
        default_factory=lambda: RoutingConfig(choose_fill="", generate="")
    )


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _normalize_type(raw: object) -> ProviderType:
    if raw in PROVIDER_PRESETS:
        return raw  # type: ignore[return-value]
    return "openai_compatible"


def _normalize_thinking(raw: object, fallback: ThinkingMode) -> ThinkingMode:
    if raw in {"off", "strip", "keep"}:
        return raw  # type: ignore[return-value]
    return fallback


def upgrade_provider_item(
    item: dict[str, Any],
    base_config: AgentConfig,
) -> dict[str, Any]:
    """旧 provider 条目补 thinking/trust_env，不覆盖已有业务字段。"""
    out = dict(item)
    ptype = _normalize_type(out.get("type") or "llamacpp")
    out["type"] = ptype
    preset = apply_preset(ptype)
    if out.get("thinking") not in {"off", "strip", "keep"}:
        out["thinking"] = preset["thinking"]
    if "trust_env" not in out:
        out["trust_env"] = preset["trust_env"]
    if not out.get("base_url"):
        out["base_url"] = str(preset.get("base_url") or base_config.base_url)
    if not out.get("model"):
        out["model"] = base_config.model
    return out


def migrate_providers(
    settings: dict[str, Any],
    base_config: AgentConfig,
) -> dict[str, Any]:
    """旧扁平模型字段 → providers[0]；已有列表则逐条升级且不丢数据。"""
    settings = dict(settings)
    if "providers" not in settings or not isinstance(settings["providers"], list):
        pid = "local"
        settings["providers"] = [
            {
                "id": pid,
                "name": "本地模型",
                "type": "llamacpp",
                "base_url": base_config.base_url,
                "api_key": base_config.api_key,
                "model": base_config.model,
            }
        ]
        settings["default_provider_id"] = pid
        settings["routing"] = {"choose_fill": pid, "generate": pid}
    upgraded: list[Any] = []
    for item in settings["providers"]:
        if isinstance(item, dict):
            upgraded.append(upgrade_provider_item(item, base_config))
        else:
            upgraded.append(item)
    settings["providers"] = upgraded
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
        ptype = _normalize_type(item.get("type", "llamacpp"))
        preset = apply_preset(ptype)
        thinking = _normalize_thinking(item.get("thinking"), preset["thinking"])
        trust_raw = item.get("trust_env")
        trust_env: bool | None
        if trust_raw is None:
            trust_env = bool(preset["trust_env"])
        else:
            trust_env = bool(trust_raw)
        providers.append(
            ProviderConfig(
                id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                type=ptype,
                base_url=str(item.get("base_url") or base_config.base_url),
                api_key=str(item.get("api_key") or ""),
                model=str(item.get("model") or base_config.model),
                sampling=item.get("sampling") if isinstance(item.get("sampling"), dict) else None,
                thinking=thinking,
                trust_env=trust_env,
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
                "thinking": p.thinking,
                "trust_env": p.trust_env,
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
        self.last_thinking: str = ""

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
        preset = apply_preset(pc.type)
        thinking: ThinkingMode = pc.thinking or preset["thinking"]
        trust_requested = (
            pc.trust_env if pc.trust_env is not None else bool(preset["trust_env"])
        )
        return OpenAICompatibleClient(
            ac,
            trust_env=trust_requested,
            thinking=thinking,
            api_key=pc.api_key,
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

    def _remember(self, client: LLMClientProtocol) -> None:
        self.last_thinking = str(getattr(client, "last_thinking", "") or "")

    def generate(self, prompt: str) -> str:
        client = self._get_client(self._config.routing.generate)
        result = client.generate(prompt)
        self._remember(client)
        return result

    def generate_int(self, prompt: str, minimum: int, maximum: int) -> int:
        client = self._get_client(self._config.routing.choose_fill)
        result = client.generate_int(prompt, minimum, maximum)
        self._remember(client)
        return result

    def generate_json(self, prompt: str, schema: dict) -> dict:
        client = self._get_client(self._config.routing.choose_fill)
        result = client.generate_json(prompt, schema)
        self._remember(client)
        return result
