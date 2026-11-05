"""A small, curated model catalog used by setup and status output.

The catalog is intentionally not a live API lookup: setup must work offline and
must be deterministic.  Unknown model ids are not an error; the wizard asks for
a context window and falls back to 128k.
"""

from __future__ import annotations

from dataclasses import dataclass

FALLBACK_CONTEXT_WINDOW = 128_000


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_window: int
    label: str = ""

    @property
    def display(self) -> str:
        name = self.label or self.id
        return f"{name} ({format_tokens(self.context_window)})"


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    key_env: str
    model_env: str
    base_env: str
    default_model: str
    default_base_url: str | None
    models: tuple[ModelInfo, ...]
    needs_base_url: bool = False

    def model(self, model_id: str) -> ModelInfo | None:
        folded = model_id.strip().lower()
        for item in self.models:
            if item.id.lower() == folded:
                return item
        return None

    def context_window(self, model_id: str) -> int | None:
        item = self.model(model_id)
        return item.context_window if item is not None else None


MOCK_PROVIDER = ProviderInfo(
    id="mock",
    label="Mock (offline)",
    key_env="",
    model_env="",
    base_env="",
    default_model="mock",
    default_base_url=None,
    models=(ModelInfo("mock", FALLBACK_CONTEXT_WINDOW, "Mock"),),
)

PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id="openai",
        label="OpenAI",
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        base_env="OPENAI_BASE_URL",
        default_model="gpt-4o-mini",
        default_base_url=None,
        models=(
            ModelInfo("gpt-4o-mini", 128_000, "GPT-4o mini"),
            ModelInfo("gpt-4o", 128_000, "GPT-4o"),
            ModelInfo("gpt-4.1-mini", 1_047_576, "GPT-4.1 mini"),
            ModelInfo("gpt-4.1", 1_047_576, "GPT-4.1"),
            ModelInfo("o4-mini", 200_000, "o4-mini"),
            ModelInfo("o3", 200_000, "o3"),
            ModelInfo("gpt-5", 400_000, "GPT-5"),
            ModelInfo("gpt-5-mini", 400_000, "GPT-5 mini"),
        ),
    ),
    ProviderInfo(
        id="deepseek",
        label="DeepSeek",
        key_env="DEEPSEEK_API_KEY",
        model_env="DEEPSEEK_MODEL",
        base_env="DEEPSEEK_BASE_URL",
        default_model="deepseek-chat",
        default_base_url="https://api.deepseek.com",
        models=(
            ModelInfo("deepseek-chat", 128_000, "DeepSeek Chat"),
            ModelInfo("deepseek-reasoner", 128_000, "DeepSeek Reasoner"),
        ),
    ),
    ProviderInfo(
        id="qwen",
        label="Qwen (DashScope)",
        key_env="DASHSCOPE_API_KEY",
        model_env="QWEN_MODEL",
        base_env="QWEN_BASE_URL",
        default_model="qwen-plus",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=(
            ModelInfo("qwen-plus", 131_072, "Qwen Plus"),
            ModelInfo("qwen-turbo", 131_072, "Qwen Turbo"),
            ModelInfo("qwen-max", 32_768, "Qwen Max"),
            ModelInfo("qwen3-max", 262_144, "Qwen3 Max"),
        ),
    ),
    ProviderInfo(
        id="custom",
        label="Custom OpenAI-compatible",
        key_env="OPENAI_API_KEY",
        model_env="SORA_CUSTOM_MODEL",
        base_env="SORA_BASE_URL",
        default_model="",
        default_base_url=None,
        models=(),
        needs_base_url=True,
    ),
)

_BY_ID = {provider.id: provider for provider in PROVIDERS}
_ALIASES = {
    "azure": "openai",
    "dashscope": "qwen",
    "ds": "deepseek",
}


def provider_info(kind: str) -> ProviderInfo | None:
    folded = str(kind or "").strip().lower()
    if folded in {"mock", "demo"}:
        return MOCK_PROVIDER
    return _BY_ID.get(_ALIASES.get(folded, folded))


def providers() -> tuple[ProviderInfo, ...]:
    return PROVIDERS


def models_for(kind: str) -> tuple[ModelInfo, ...]:
    info = provider_info(kind)
    return info.models if info is not None else ()


def default_model(kind: str) -> str:
    info = provider_info(kind)
    return info.default_model if info is not None else ""


def default_base_url(kind: str) -> str | None:
    info = provider_info(kind)
    return info.default_base_url if info is not None else None


def key_env(kind: str) -> str:
    info = provider_info(kind)
    return info.key_env if info is not None else "OPENAI_API_KEY"


def model_env(kind: str) -> str:
    info = provider_info(kind)
    return info.model_env if info is not None else "SORA_CUSTOM_MODEL"


def base_env(kind: str) -> str:
    info = provider_info(kind)
    return info.base_env if info is not None else "SORA_BASE_URL"


def context_window_for(kind: str, model: str) -> int | None:
    """Return the known context window, or ``None`` for an unknown model."""
    info = provider_info(kind)
    return info.context_window(model) if info is not None else None


def format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        value = tokens / 1_000_000
        return f"{value:.4g}M"
    if tokens >= 1_000:
        value = tokens / 1_000
        return f"{value:.4g}k"
    return f"{tokens:,}"
