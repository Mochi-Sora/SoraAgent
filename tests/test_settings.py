"""Tests for the persisted settings schema, presets, and secret storage."""

from __future__ import annotations

import json
import stat

import pytest

from sora_agent.model_catalog import (
    FALLBACK_CONTEXT_WINDOW,
    context_window_for,
    format_tokens,
    provider_info,
)
from sora_agent.persona import apply_persona, display_name, persona_prompt
from sora_agent.settings import (
    AgentSettings,
    ChannelSettings,
    FileSettings,
    MemorySettings,
    PersonalitySettings,
    ProviderSettings,
    Settings,
    SettingsError,
    advanced_settings,
    config_path,
    env_flag,
    env_float,
    env_int,
    legacy_settings,
    load_secrets,
    load_settings,
    recommended_settings,
    resolved,
    save_secrets,
    save_settings,
    secrets_path,
    update_secrets,
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_recommended_preset_turns_everything_on():
    settings = recommended_settings()

    assert settings.setup.mode == "recommended"
    assert settings.agent.name == "Sora"
    assert settings.agent.icon == "☁️"
    assert settings.agent.personality.base == "friendly"
    assert settings.agent.personality.warmth == "very_warm"
    assert settings.provider.temperature == 0.7

    features = settings.features
    assert features.memory.enabled is True
    assert features.memory.sidecar is True
    assert features.memory.auto_download is True
    assert features.memory.dream is True
    assert features.memory.backup is True
    assert features.memory.wake is True
    assert features.files.enabled is True
    assert features.identity is True
    assert features.clock is True
    assert features.calendar is True
    assert features.sessions is True
    assert features.tools == "demo"
    assert features.approval == "ask"
    assert features.quality is True

    assert settings.limits.max_steps == 12
    assert settings.limits.max_turns == 80
    assert settings.limits.provider_timeout == 60.0
    assert settings.limits.provider_retries == 3


def test_advanced_preset_turns_optional_features_off():
    settings = advanced_settings()

    assert settings.setup.mode == "advanced"
    assert settings.provider.temperature == 0.1
    assert settings.agent.personality.base == "neutral"
    assert settings.agent.personality.warmth == "neutral"

    features = settings.features
    assert features.memory.enabled is False
    assert features.files.enabled is False
    assert features.identity is False
    assert features.clock is False
    assert features.calendar is False
    assert features.sessions is False
    assert features.tools == "none"
    assert features.approval == "off"
    assert features.quality is False


def test_legacy_preset_matches_the_environment_only_gateway():
    settings = legacy_settings()

    assert settings.setup.mode == "legacy"
    assert settings.provider.kind == "mock"
    assert settings.features.memory.enabled is False
    assert settings.features.identity is False
    assert settings.features.clock is True
    assert settings.features.calendar is True
    assert settings.features.sessions is True
    assert settings.features.tools == "none"
    assert settings.features.approval == "ask"
    assert settings.features.quality is True
    assert settings.limits.max_steps == 8
    assert settings.limits.max_turns == 40


# ---------------------------------------------------------------------------
# Parsing and round-tripping
# ---------------------------------------------------------------------------


def test_file_settings_validate_limits():
    with pytest.raises(SettingsError):
        FileSettings(max_file_bytes=0)
    with pytest.raises(SettingsError):
        FileSettings(max_chars=100)
    with pytest.raises(SettingsError):
        FileSettings(max_chars=1000, total_max_chars=500)

    settings = FileSettings(max_chars=1000, total_max_chars=2000)
    assert settings.max_chars == 1000
    assert settings.total_max_chars == 2000


def test_settings_round_trip_preserves_values():
    original = recommended_settings()
    original.agent.name = "Nimbus"
    original.features.memory.recall_top_k = 3

    loaded = Settings.from_dict(json.loads(json.dumps(original.to_dict())))

    assert loaded.to_dict() == original.to_dict()
    assert loaded.agent.name == "Nimbus"
    assert loaded.features.memory.recall_top_k == 3


def test_partial_config_inherits_the_preset_it_claims():
    loaded = Settings.from_dict(
        {
            "setup": {"mode": "advanced"},
            "provider": {"temperature": 0.5},
        }
    )

    assert loaded.setup.mode == "advanced"
    assert loaded.provider.temperature == 0.5
    # Missing sections inherit the Advanced preset, not built-in dataclass
    # defaults (which would silently enable safety features).
    assert loaded.features.memory.enabled is False
    assert loaded.features.approval == "off"
    assert loaded.features.quality is False
    assert loaded.limits.max_steps == 8


def test_partial_recommended_config_keeps_recommended_defaults():
    loaded = Settings.from_dict(
        {
            "setup": {"mode": "recommended"},
            "agent": {"name": "Mochi"},
        }
    )

    assert loaded.agent.name == "Mochi"
    assert loaded.agent.icon == "☁️"
    assert loaded.features.memory.enabled is True
    assert loaded.features.approval == "ask"
    assert loaded.features.quality is True


def test_unknown_setup_mode_is_rejected():
    with pytest.raises(SettingsError):
        Settings.from_dict({"setup": {"mode": "quickstart"}})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PersonalitySettings(base="grumpy", warmth="neutral"),
        lambda: PersonalitySettings(base="friendly", warmth="boiling"),
        lambda: ProviderSettings(temperature=2.5),
        lambda: ProviderSettings(context_window=10),
        lambda: ChannelSettings(parse_mode="BBCode"),
    ],
)
def test_invalid_settings_values_are_rejected(factory):
    with pytest.raises(SettingsError):
        factory()


def test_feature_validation_rejects_unknown_policy_values():
    with pytest.raises(SettingsError):
        Settings.from_dict(
            {"setup": {"mode": "advanced"}, "features": {"tools": "shell"}}
        )
    with pytest.raises(SettingsError):
        Settings.from_dict(
            {"setup": {"mode": "advanced"}, "features": {"approval": "maybe"}}
        )


def test_agent_name_falls_back_and_icon_can_be_hidden():
    agent = AgentSettings(name="  ", icon="")

    assert agent.name == "Sora"
    assert agent.icon == ""


def test_memory_from_dict_defaults_do_not_enable_disabled_memory():
    memory = MemorySettings.from_dict({"enabled": False})

    assert memory.enabled is False
    assert memory.sidecar is True


# ---------------------------------------------------------------------------
# Files, paths, and secrets
# ---------------------------------------------------------------------------


def test_config_path_uses_environment_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SORA_CONFIG", str(tmp_path / "custom.json"))
    assert config_path() == tmp_path / "custom.json"
    assert config_path(tmp_path / "explicit.json") == tmp_path / "explicit.json"

    monkeypatch.delenv("SORA_CONFIG")
    assert config_path().name == "config.json"
    assert config_path().parent.name == ".sora_agent"


def test_secrets_path_uses_environment_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SORA_SECRETS", str(tmp_path / "custom.env"))
    assert secrets_path() == tmp_path / "custom.env"
    assert secrets_path(tmp_path / "explicit.env") == tmp_path / "explicit.env"


def test_save_and_load_settings_are_atomic_and_private(tmp_path):
    target = tmp_path / "nested" / "config.json"
    settings = recommended_settings()
    settings.agent.name = "Mochi"

    saved = save_settings(settings, target)
    loaded = load_settings(target)

    assert saved == target
    assert loaded is not None
    assert loaded.agent.name == "Mochi"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_save_settings_validates_before_writing(tmp_path):
    settings = advanced_settings()
    settings.provider.temperature = 3.0
    target = tmp_path / "config.json"

    with pytest.raises(SettingsError):
        save_settings(settings, target)

    assert not target.exists()


def test_missing_config_returns_none(tmp_path):
    assert load_settings(tmp_path / "absent.json") is None


def test_invalid_json_reports_the_path(tmp_path):
    target = tmp_path / "config.json"
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(SettingsError, match="not valid JSON"):
        load_settings(target)


def test_secrets_round_trip_quoting_and_comments(tmp_path):
    target = tmp_path / "secrets.env"
    save_secrets(
        {
            "PLAIN": "value",
            "SPACED": "two words",
            "HASH": "value # not a comment",
            "QUOTED": 'say "hi"',
        },
        target,
    )

    values = load_secrets(target)

    assert values == {
        "PLAIN": "value",
        "SPACED": "two words",
        "HASH": "value # not a comment",
        "QUOTED": 'say "hi"',
    }
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert "PLAIN=value" in target.read_text(encoding="utf-8")


def test_secrets_parser_accepts_export_and_comments(tmp_path):
    target = tmp_path / "secrets.env"
    target.write_text(
        "# comment\n"
        "export OPENAI_API_KEY=sk-test\n"
        "\n"
        "IGNORED_LINE\n"
        "QUOTED='single quoted'\n",
        encoding="utf-8",
    )

    assert load_secrets(target) == {
        "OPENAI_API_KEY": "sk-test",
        "QUOTED": "single quoted",
    }


def test_update_secrets_adds_and_removes_keys(tmp_path):
    target = tmp_path / "secrets.env"
    save_secrets({"KEEP": "yes", "REMOVE": "me"}, target)

    merged = update_secrets({"REMOVE": "", "NEW": "value"}, target)

    assert merged == {"KEEP": "yes", "NEW": "value"}
    assert load_secrets(target) == {"KEEP": "yes", "NEW": "value"}


def test_secrets_reject_newlines(tmp_path):
    with pytest.raises(SettingsError):
        save_secrets({"TOKEN": "line one\nline two"}, tmp_path / "secrets.env")


# ---------------------------------------------------------------------------
# Precedence helpers
# ---------------------------------------------------------------------------


def test_resolved_returns_the_first_explicit_value():
    assert resolved(None, None, 0) == 0
    assert resolved(None, "env") == "env"
    assert resolved(None, None) is None


def test_environment_helpers(monkeypatch):
    monkeypatch.setenv("SORA_TEST_FLAG", "yes")
    monkeypatch.setenv("SORA_TEST_INT", "42")
    monkeypatch.setenv("SORA_TEST_FLOAT", "1.5")
    monkeypatch.setenv("SORA_TEST_BAD", "nope")
    monkeypatch.setenv("SORA_TEST_EMPTY", "")

    assert env_flag("SORA_TEST_FLAG") is True
    assert env_flag("SORA_TEST_MISSING") is None
    assert env_int("SORA_TEST_INT") == 42
    assert env_int("SORA_TEST_BAD") is None
    assert env_float("SORA_TEST_FLOAT") == 1.5
    assert env_float("SORA_TEST_BAD") is None
    assert env_int("SORA_TEST_EMPTY") is None


# ---------------------------------------------------------------------------
# Model catalog and persona
# ---------------------------------------------------------------------------


def test_model_catalog_knows_context_windows_and_aliases():
    assert context_window_for("openai", "gpt-5.4-mini") == 272_000
    assert context_window_for("openai", "gpt-6.1-sol") == 922_000
    assert context_window_for("openai", "gpt-4o-mini") == 128_000
    assert context_window_for("openai", "gpt-4.1") == 1_047_576
    assert context_window_for("deepseek", "deepseek-flash") == 1_000_000
    assert context_window_for("deepseek", "deepseek-chat") == 131_072
    assert context_window_for("ds", "deepseek-reasoner") == 131_072
    assert context_window_for("qwen", "qwen3.7-plus") == 991_808
    assert context_window_for("dashscope", "qwen3.8-max") == 991_808
    assert context_window_for("openai", "some-future-model") is None
    dashscope = provider_info("dashscope")
    assert dashscope is not None
    assert dashscope.id == "qwen"
    mock = provider_info("mock")
    assert mock is not None
    assert mock.label == "Mock (offline)"
    assert mock.key_env == ""
    assert context_window_for("mock", "mock") == FALLBACK_CONTEXT_WINDOW


def test_model_catalog_fallback_is_documented_and_positive():
    assert FALLBACK_CONTEXT_WINDOW == 128_000
    assert format_tokens(1_047_576) == "1.048M"
    assert format_tokens(128_000) == "128k"
    assert format_tokens(999) == "999"


def test_persona_prompt_mentions_the_name_and_tone():
    prompt = persona_prompt("Nimbus", "friendly", "very_warm")

    assert "Your name is Nimbus." in prompt
    assert "warm" in prompt.lower()


def test_neutral_persona_is_just_the_name():
    assert persona_prompt("Sora", "neutral", "neutral") == "Your name is Sora."


def test_apply_persona_appends_guidance_to_a_system_prompt():
    combined = apply_persona("Base prompt.", "Sora", "friendly", "warm")

    assert combined.startswith("Base prompt.")
    assert "Your name is Sora." in combined


def test_display_name_combines_icon_and_name():
    agent = AgentSettings(name="Nimbus", icon="⛅")

    assert display_name(agent) == "⛅ Nimbus"
    assert display_name(AgentSettings(name="Nimbus", icon="")) == "Nimbus"
