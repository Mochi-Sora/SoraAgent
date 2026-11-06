"""Tests for settings-driven runtime construction and precedence."""

from __future__ import annotations

import pytest

from sora_agent.memory.config import MemoryConfig
from sora_agent.providers import MockProvider
from sora_agent.runtime_setup import (
    approval_mode,
    build_calendar,
    build_identity,
    build_memory,
    build_provider,
    build_registry,
    build_sessions,
    clock_enabled,
    context_window,
    feature_enabled,
    load_runtime,
    memory_enabled,
    provider_kind,
    quality_enabled,
    resolve_api_key,
    streaming_enabled,
)
from sora_agent.settings import (
    MemorySettings,
    advanced_settings,
    recommended_settings,
    save_settings,
)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_runtime_without_a_config_uses_legacy_defaults(tmp_path):
    settings, secrets, configured = load_runtime(
        tmp_path / "missing.json", tmp_path / "missing.env"
    )

    assert configured is False
    assert settings.setup.mode == "legacy"
    assert settings.provider.kind == "mock"
    assert settings.features.memory.enabled is False
    assert secrets == {}


def test_load_runtime_reads_a_saved_config_and_secrets(tmp_path):
    config = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.env"
    settings = recommended_settings()
    settings.agent.name = "Nimbus"
    save_settings(settings, config)
    secrets_path.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

    loaded, secrets, configured = load_runtime(config, secrets_path)

    assert configured is True
    assert loaded.agent.name == "Nimbus"
    assert secrets == {"OPENAI_API_KEY": "sk-test"}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_provider_kind_uses_environment_then_settings(monkeypatch):
    settings = recommended_settings()
    settings.provider.kind = "deepseek"

    monkeypatch.delenv("SORA_PROVIDER", raising=False)
    assert provider_kind(settings) == "deepseek"

    monkeypatch.setenv("SORA_PROVIDER", "qwen")
    assert provider_kind(settings) == "qwen"


def test_build_provider_honours_explicit_values_environment_and_settings(
    monkeypatch,
):
    settings = recommended_settings()
    settings.provider.kind = "openai"
    settings.provider.model = "config-model"
    settings.provider.temperature = 0.4

    monkeypatch.delenv("SORA_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SORA_TEMPERATURE", raising=False)

    configured = build_provider(settings, api_key="key")
    assert configured.model == "config-model"
    assert configured.temperature == 0.4

    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("SORA_TEMPERATURE", "1.2")
    from_env = build_provider(settings, api_key="key")
    assert from_env.model == "env-model"
    assert from_env.temperature == 1.2

    explicit = build_provider(
        settings, api_key="key", model="cli-model", temperature=0.0
    )
    assert explicit.model == "cli-model"
    assert explicit.temperature == 0.0


def test_build_provider_reads_a_key_from_secrets(monkeypatch):
    settings = recommended_settings()
    settings.provider.kind = "openai"

    monkeypatch.delenv("SORA_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = build_provider(settings, {"OPENAI_API_KEY": "sk-from-file"})

    assert provider.api_key == "sk-from-file"


def test_build_provider_environment_key_wins_over_secrets(monkeypatch):
    settings = recommended_settings()
    settings.provider.kind = "openai"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    provider = build_provider(settings, {"OPENAI_API_KEY": "sk-from-file"})

    assert provider.api_key == "sk-from-env"


def test_build_provider_supports_mock_and_custom(monkeypatch):
    mock_settings = advanced_settings()
    mock_settings.provider.kind = "mock"
    monkeypatch.delenv("SORA_PROVIDER", raising=False)
    assert isinstance(build_provider(mock_settings), MockProvider)

    custom = advanced_settings()
    custom.provider.kind = "custom"
    custom.provider.base_url = "https://example.test/v1"
    provider = build_provider(custom, api_key="key")
    assert provider.base_url == "https://example.test/v1"

    custom.provider.base_url = None
    with pytest.raises(ValueError, match="base_url"):
        build_provider(custom, api_key="key")


def test_resolve_api_key_uses_environment_before_secrets(monkeypatch):
    settings = recommended_settings()
    settings.provider.kind = "deepseek"

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert resolve_api_key(settings, {"DEEPSEEK_API_KEY": "file"}) == "file"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "env")
    assert resolve_api_key(settings, {"DEEPSEEK_API_KEY": "file"}) == "env"


# ---------------------------------------------------------------------------
# Feature switches and limits
# ---------------------------------------------------------------------------


def test_feature_switch_uses_environment_before_settings(monkeypatch):
    settings = advanced_settings()
    settings.features.quality = True

    monkeypatch.delenv("SORA_QUALITY", raising=False)
    assert feature_enabled(settings, "quality", "SORA_QUALITY") is True

    monkeypatch.setenv("SORA_QUALITY", "0")
    assert feature_enabled(settings, "quality", "SORA_QUALITY") is False


def test_switching_settings_does_not_override_the_environment(monkeypatch):
    settings = recommended_settings()

    monkeypatch.setenv("SORA_MEMORY", "0")
    assert memory_enabled(settings) is False
    monkeypatch.setenv("SORA_MEMORY", "1")
    assert memory_enabled(settings) is True

    monkeypatch.setenv("SORA_QUALITY", "off")
    assert quality_enabled(settings) is False
    monkeypatch.setenv("SORA_STREAMING", "off")
    assert streaming_enabled(settings) is False
    monkeypatch.setenv("SORA_CLOCK", "off")
    assert clock_enabled(settings) is False


def test_approval_mode_is_env_overridable(monkeypatch):
    settings = recommended_settings()

    monkeypatch.delenv("SORA_APPROVAL", raising=False)
    assert approval_mode(settings) == "ask"

    monkeypatch.setenv("SORA_APPROVAL", "off")
    assert approval_mode(settings) == "off"

    settings.features.approval = "off"
    monkeypatch.delenv("SORA_APPROVAL", raising=False)
    assert approval_mode(settings) == "off"


def test_context_window_uses_environment_override_and_a_minimum(monkeypatch):
    settings = recommended_settings()
    settings.provider.context_window = 32_000

    monkeypatch.delenv("SORA_CONTEXT_WINDOW", raising=False)
    assert context_window(settings) == 32_000

    monkeypatch.setenv("SORA_CONTEXT_WINDOW", "500")
    assert context_window(settings) == 1_000


# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------


def test_identity_respects_settings_and_environment(monkeypatch, tmp_path):
    settings = recommended_settings()
    settings.features.identity = False
    monkeypatch.delenv("SORA_IDENTITY", raising=False)
    assert build_identity(settings) is None

    settings.features.identity = True
    monkeypatch.setenv("SORA_IDENTITY", "0")
    assert build_identity(settings) is None

    monkeypatch.delenv("SORA_IDENTITY", raising=False)
    settings.paths.identity_root = str(tmp_path / "identity")
    store = build_identity(settings)
    assert store is not None
    assert store.config.root == tmp_path / "identity"


def test_registry_can_be_empty_or_the_demo_pack(monkeypatch, tmp_path):
    settings = advanced_settings()
    monkeypatch.delenv("SORA_TOOLS", raising=False)
    assert build_registry(settings).all() == []

    settings.features.tools = "demo"
    names = {tool.name for tool in build_registry(settings).all()}
    assert "lookup_order" in names


def test_sessions_can_be_disabled_or_pointed_at_a_directory(monkeypatch, tmp_path):
    settings = recommended_settings()
    monkeypatch.delenv("SORA_SESSIONS", raising=False)
    monkeypatch.delenv("SORA_SESSIONS_ROOT", raising=False)

    settings.features.sessions = False
    assert build_sessions(settings) is None

    settings.features.sessions = True
    settings.paths.sessions_root = str(tmp_path / "sessions")
    store = build_sessions(settings)
    assert store is not None
    assert store.root == tmp_path / "sessions"


def test_calendar_can_be_disabled(monkeypatch):
    settings = advanced_settings()
    monkeypatch.delenv("SORA_CALENDAR", raising=False)
    assert build_calendar(settings) is False

    settings.features.calendar = True
    store = build_calendar(settings)
    assert store is not None


def test_memory_settings_reach_the_memory_config(monkeypatch, tmp_path):
    monkeypatch.delenv("SORA_MEMORY", raising=False)
    monkeypatch.delenv("SORA_MEMORY_RECALL_TOP_K", raising=False)
    monkeypatch.delenv("SORA_MEMORY_RECALL_MAX_CHARS", raising=False)
    monkeypatch.delenv("SORA_MEMORY_DREAM_AUTO", raising=False)

    config = MemoryConfig.from_settings(
        {
            "enabled": True,
            "recall_top_k": 4,
            "recall_max_chars": 900,
            "retrieval_mode": "lexical",
            "dream": False,
        },
        root=tmp_path / "memory",
    )

    assert config.enabled is True
    assert config.root == tmp_path / "memory"
    assert config.recall_top_k == 4
    assert config.recall_max_chars == 900
    assert config.recall_retrieval_mode == "lexical"
    assert config.dream_auto is False


def test_memory_environment_wins_over_saved_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SORA_MEMORY", "1")
    monkeypatch.setenv("SORA_MEMORY_RECALL_TOP_K", "2")
    monkeypatch.delenv("SORA_MEMORY_DREAM_AUTO", raising=False)

    config = MemoryConfig.from_settings(
        {"enabled": True, "recall_top_k": 9, "dream": False},
        root=tmp_path / "memory",
    )

    assert config.enabled is True
    assert config.recall_top_k == 2


def test_disabled_memory_builds_no_hook(monkeypatch, tmp_path):
    settings = advanced_settings()
    settings.paths.memory_root = str(tmp_path / "memory")
    monkeypatch.setenv("SORA_MEMORY", "0")

    assert build_memory(settings, identity=None, secrets={}) is None


def test_memory_hook_uses_the_configured_root(monkeypatch, tmp_path):
    settings = recommended_settings()
    settings.features.memory = MemorySettings(
        enabled=True,
        sidecar=False,
        auto_download=False,
        dream=False,
        backup=False,
        wake=False,
    )
    settings.paths.memory_root = str(tmp_path / "memory")
    monkeypatch.setenv("SORA_MEMORY", "1")
    monkeypatch.delenv("SORA_MEMORY_DREAM_AUTO", raising=False)

    hook = build_memory(settings, identity=None, secrets={})

    assert hook is not None
    try:
        assert hook.config.root == tmp_path / "memory"
        assert hook.config.enabled is True
        assert hook.config.wake_model == ""
    finally:
        hook.close()
