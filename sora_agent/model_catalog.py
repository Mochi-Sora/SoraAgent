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

# Curated from provider documentation in November 2026.  Context windows are
# input limits where the provider publishes a separate input budget.
PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id="openai",
        label="OpenAI",
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        base_env="OPENAI_BASE_URL",
        default_model="gpt-5.4-mini",
        default_base_url=None,
        models=(
            ModelInfo("gpt-6.1-sol", 922_000, "GPT-6.1 Sol"),
            ModelInfo("gpt-6.1-terra", 922_000, "GPT-6.1 Terra"),
            ModelInfo("gpt-6-astra", 922_000, "GPT-6 Astra"),
            ModelInfo("gpt-6-sol", 922_000, "GPT-6 Sol"),
            ModelInfo("gpt-5.6", 922_000, "GPT-5.6"),
            ModelInfo("gpt-5.5", 922_000, "GPT-5.5"),
            ModelInfo("gpt-5.4", 922_000, "GPT-5.4"),
            ModelInfo("gpt-5.4-mini", 272_000, "GPT-5.4 mini"),
            ModelInfo("gpt-5.4-nano", 272_000, "GPT-5.4 nano"),
            ModelInfo("gpt-5.2", 272_000, "GPT-5.2"),
            ModelInfo("gpt-4.1", 1_047_576, "GPT-4.1"),
            ModelInfo("gpt-4o", 128_000, "GPT-4o"),
            ModelInfo("gpt-4o-mini", 128_000, "GPT-4o mini (legacy)"),
            ModelInfo("o3", 200_000, "o3"),
            ModelInfo("o4-mini", 200_000, "o4-mini"),
        ),
    ),
    ProviderInfo(
        id="deepseek",
        label="DeepSeek",
        key_env="DEEPSEEK_API_KEY",
        model_env="DEEPSEEK_MODEL",
        base_env="DEEPSEEK_BASE_URL",
        default_model="deepseek-flash",
        default_base_url="https://api.deepseek.com",
        models=(
            ModelInfo("deepseek-flash", 1_000_000, "DeepSeek Flash"),
            ModelInfo("deepseek-v4-pro", 1_000_000, "DeepSeek V4 Pro"),
            ModelInfo(
                "deepseek-v4-flash",
                1_000_000,
                "DeepSeek V4 Flash (legacy alias)",
            ),
            ModelInfo("deepseek-chat", 131_072, "DeepSeek Chat (legacy)"),
            ModelInfo(
                "deepseek-reasoner", 131_072, "DeepSeek Reasoner (legacy)"
            ),
        ),
    ),
    ProviderInfo(
        id="qwen",
        label="Qwen (DashScope)",
        key_env="DASHSCOPE_API_KEY",
        model_env="QWEN_MODEL",
        base_env="QWEN_BASE_URL",
        default_model="qwen3.7-plus",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=(
            ModelInfo("qwen3.8-max", 991_808, "Qwen3.8 Max"),
            ModelInfo("qwen3.7-plus", 991_808, "Qwen3.7 Plus"),
            ModelInfo("qwen3.8-flash", 991_808, "Qwen3.8 Flash"),
            ModelInfo("qwen3.8-omni-flash", 991_808, "Qwen3.8 Omni Flash"),
            ModelInfo("qwen3-coder-plus", 997_952, "Qwen3 Coder Plus"),
            ModelInfo("qwen3-vl-plus", 260_096, "Qwen3 VL Plus"),
            ModelInfo("qwen3-max", 258_048, "Qwen3 Max"),
            ModelInfo("qwen-plus", 129_024, "Qwen Plus (legacy)"),
            ModelInfo("qwen-turbo", 129_024, "Qwen Turbo (legacy)"),
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
