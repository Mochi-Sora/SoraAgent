"""Build runtime objects from settings, preserving environment precedence.

Both ``sora-gateway`` and the new ``sora`` CLI use this module, so a saved
configuration means the same thing everywhere.  Keep it standard-library-only
and import optional subsystems lazily.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .model_catalog import (
    FALLBACK_CONTEXT_WINDOW,
    base_env,
    default_base_url,
    default_model,
    key_env,
    model_env,
    provider_info,
)
from .providers import default_provider
from .settings import (
    Settings,
    env_flag,
    env_float,
    env_int,
    legacy_settings,
    load_secrets,
    load_settings,
)

_OPENAI_COMPATIBLE = frozenset({"openai", "azure", "deepseek", "ds", "qwen", "dashscope", "custom"})


def load_runtime(
    path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> tuple[Settings, dict[str, str], bool]:
    """Return ``(settings, secrets, configured)``.

    With no config file the legacy gateway defaults are returned and
    ``configured`` is ``False``.  That is what keeps an environment-only
    install behaving exactly as before.
    """
    settings = load_settings(path)
    configured = settings is not None
    return settings or legacy_settings(), load_secrets(secrets_path), configured


def provider_kind(settings: Settings) -> str:
    explicit = os.getenv("SORA_PROVIDER", "").strip()
    if explicit:
        return explicit.lower()
    return (settings.provider.kind or "mock").lower()


def build_provider(
    settings: Settings,
    secrets: Mapping[str, str] | None = None,
    *,
    kind: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> Any:
    """Build the configured provider with env overrides applied."""
    secrets = secrets or {}
    env_kind = os.getenv("SORA_PROVIDER", "").strip().lower()
    resolved_kind = (kind or env_kind or settings.provider.kind or "mock").lower()
    same_as_config = resolved_kind == (settings.provider.kind or "").lower()

    catalog_kind = resolved_kind
    if resolved_kind in {"azure"}:
        catalog_kind = "openai"
    elif resolved_kind in {"ds"}:
        catalog_kind = "deepseek"
    elif resolved_kind in {"dashscope"}:
        catalog_kind = "qwen"

    resolved_model = (
        model
        or os.getenv(model_env(catalog_kind), "").strip()
        or (settings.provider.model if same_as_config else "")
        or default_model(catalog_kind)
    )
    resolved_base = (
        base_url
        or os.getenv(base_env(catalog_kind), "").strip()
        or (settings.provider.base_url if same_as_config else None)
        or default_base_url(catalog_kind)
    )
    resolved_key = (
        api_key
        or os.getenv(key_env(catalog_kind), "").strip()
        or secrets.get(key_env(catalog_kind), "").strip()
        or None
    )
    env_temperature = env_float("SORA_TEMPERATURE")
    resolved_temperature = (
        temperature
        if temperature is not None
        else (
            env_temperature
            if env_temperature is not None
            else settings.provider.temperature
        )
    )
    env_timeout = env_float("SORA_PROVIDER_TIMEOUT")
    resolved_timeout = (
        timeout
        if timeout is not None
        else (
            env_timeout
            if env_timeout is not None
            else settings.limits.provider_timeout
        )
    )

    if resolved_kind in {"mock", "demo"}:
        return default_provider("mock", timeout=resolved_timeout)
    return default_provider(
        resolved_kind,
        timeout=resolved_timeout,
        model=resolved_model or None,
        api_key=resolved_key,
        base_url=resolved_base or None,
        temperature=resolved_temperature,
    )


def context_window(settings: Settings) -> int:
    override = env_int("SORA_CONTEXT_WINDOW")
    value = override if override is not None else settings.provider.context_window
    return max(1_000, int(value or FALLBACK_CONTEXT_WINDOW))


def provider_label(settings: Settings) -> str:
    kind = provider_kind(settings)
    info = provider_info(kind)
    return info.label if info is not None else kind


def openai_compatible_base(settings: Settings) -> str | None:
    """Return the base URL the wake reviewer should call, if any."""
    kind = provider_kind(settings)
    if kind not in _OPENAI_COMPATIBLE:
        return None
    catalog_kind = {
        "azure": "openai",
        "ds": "deepseek",
        "dashscope": "qwen",
    }.get(kind, kind)
    return (
        os.getenv(base_env(catalog_kind), "").strip()
        or settings.provider.base_url
        or default_base_url(catalog_kind)
    )


def resolve_api_key(
    settings: Settings, secrets: Mapping[str, str] | None = None
) -> str:
    secrets = secrets or {}
    kind = provider_kind(settings)
    catalog_kind = {
        "azure": "openai",
        "ds": "deepseek",
        "dashscope": "qwen",
    }.get(kind, kind)
    env_name = key_env(catalog_kind)
    return (
        os.getenv(env_name, "").strip()
        or secrets.get(env_name, "").strip()
        or ""
    )


def build_identity(settings: Settings, *, enabled: bool | None = None) -> Any:
    """Build the identity store, or return ``None`` when disabled."""
    from .identity import default_identity

    return default_identity(
        enabled=(
            bool(settings.features.identity) if enabled is None else enabled
        ),
        root=settings.paths.identity_root,
    )


def build_calendar(settings: Settings) -> Any:
    if not feature_enabled(settings, "calendar", "SORA_CALENDAR"):
        return False
    from .calendar import default_calendar

    path = os.getenv("SORA_CALENDAR_DB", "").strip() or settings.paths.calendar_db
    return default_calendar(path)


def build_sessions(settings: Settings) -> Any:
    if not feature_enabled(settings, "sessions", "SORA_SESSIONS"):
        return None
    from .channels.sessions import SessionStore

    configured = os.getenv("SORA_SESSIONS_ROOT", "").strip()
    root = configured or settings.paths.sessions_root
    return SessionStore(Path(root).expanduser() if root else None)


def build_registry(settings: Settings) -> Any:
    from .tools import ToolRegistry

    kind = (os.getenv("SORA_TOOLS", "").strip() or settings.features.tools or "none").lower()
    if kind == "demo":
        from .tools import registry as demo_tools

        return demo_tools.copy()
    return ToolRegistry()


def build_memory(
    settings: Settings,
    identity: Any,
    secrets: Mapping[str, str] | None = None,
) -> Any:
    """Build the memory hook from settings, or return ``None``."""
    from .memory.config import MemoryConfig
    from .memory.hook import MemoryHook

    secrets = secrets or {}
    memory = settings.features.memory
    root = os.getenv("SORA_MEMORY_ROOT", "").strip() or settings.paths.memory_root

    provider = provider_kind(settings)
    wake_model = ""
    wake_base_url = ""
    wake_api_key = ""
    if memory.wake and provider in _OPENAI_COMPATIBLE:
        wake_model = settings.provider.model
        wake_base_url = openai_compatible_base(settings) or ""
        wake_api_key = resolve_api_key(settings, secrets)

    config = MemoryConfig.from_settings(
        memory.__dict__,
        root=root,
        wake_model=wake_model,
        wake_base_url=wake_base_url,
        wake_api_key=wake_api_key,
    )
    if not config.enabled:
        return None
    return MemoryHook(
        config,
        identity=identity,
        auto_dream=config.dream_auto,
    )


def approval_mode(settings: Settings) -> str:
    override = os.getenv("SORA_APPROVAL", "").strip().lower()
    if override in {"off", "ask"}:
        return override
    mode = str(settings.features.approval or "ask").lower()
    return mode if mode in {"off", "ask"} else "ask"


def quality_enabled(settings: Settings) -> bool:
    return feature_enabled(settings, "quality", "SORA_QUALITY")


def streaming_enabled(settings: Settings) -> bool:
    return feature_enabled(settings, "streaming", "SORA_STREAMING")


def clock_enabled(settings: Settings) -> bool:
    return feature_enabled(settings, "clock", "SORA_CLOCK")


def feature_enabled(settings: Settings, name: str, env_name: str) -> bool:
    override = env_flag(env_name)
    if override is not None:
        return override
    return bool(getattr(settings.features, name))


def memory_enabled(settings: Settings) -> bool:
    override = env_flag("SORA_MEMORY")
    if override is not None:
        return override
    return bool(settings.features.memory.enabled)
