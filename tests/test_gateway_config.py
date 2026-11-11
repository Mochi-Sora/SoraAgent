"""Tests for the config-aware channel gateway and its precedence rules."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sora_agent.channels import cli as gateway
from sora_agent.channels.cli import main as gateway_main
from sora_agent.settings import (
    advanced_settings,
    legacy_settings,
    recommended_settings,
    save_settings,
)


def _clear_runtime_overrides(monkeypatch):
    for name in (
        "SORA_PROVIDER",
        "SORA_TEMPERATURE",
        "SORA_CONTEXT_WINDOW",
        "SORA_APPROVAL",
        "SORA_QUALITY",
        "SORA_CLOCK",
        "SORA_CALENDAR",
        "SORA_SESSIONS",
        "SORA_SESSIONS_ROOT",
        "SORA_MEMORY",
        "SORA_IDENTITY",
        "SORA_TELEGRAM_TOKEN",
        "SORA_TELEGRAM_ALLOW_FROM",
        "SORA_TELEGRAM_PARSE_MODE",
        "SORA_TELEGRAM_DOWNLOAD_FILES",
        "SORA_TELEGRAM_MAX_ATTACHMENT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def _capture_serve(monkeypatch):
    captured: dict = {}

    @dataclass
    class FakeConfig:
        allow_from: tuple[str, ...] = ()
        download_attachments: bool = True

    class FakePlugin:
        def load_channel_class(self):
            return type("FakeChannel", (), {})

    async def fake_serve(channel_cls, config, **kwargs):
        captured["channel_cls"] = channel_cls
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setattr(gateway, "_serve", fake_serve)
    monkeypatch.setattr(gateway, "load_plugin", lambda name: FakePlugin())
    monkeypatch.setattr(
        gateway,
        "build_channel_config",
        lambda name, settings, secrets, args: FakeConfig(
            allow_from=tuple(settings.channel.allow_from)
        ),
    )
    return captured


# ---------------------------------------------------------------------------
# Telegram configuration precedence
# ---------------------------------------------------------------------------


def test_telegram_config_uses_saved_values(monkeypatch):
    _clear_runtime_overrides(monkeypatch)
    settings = recommended_settings()
    settings.channel.allow_from = ("11", "22")
    settings.channel.parse_mode = "HTML"
    settings.channel.api_base = "https://telegram.example.test"
    settings.channel.poll_timeout = 10
    settings.channel.request_timeout = 12.5
    settings.channel.max_message_chars = 2048

    config = gateway._telegram_config(
        settings,
        {"SORA_TELEGRAM_TOKEN": "saved-token"},
        SimpleNamespace(token=None, allow=None),
    )

    assert config.token == "saved-token"
    assert config.allow_from == ("11", "22")
    assert config.parse_mode == "HTML"
    assert config.api_base == "https://telegram.example.test"
    assert config.poll_timeout == 10
    assert config.request_timeout == 12.5
    assert config.max_message_chars == 2048


def test_telegram_environment_overrides_saved_values(monkeypatch):
    _clear_runtime_overrides(monkeypatch)
    settings = recommended_settings()
    settings.channel.allow_from = ("11",)
    settings.channel.parse_mode = "HTML"
    settings.channel.poll_timeout = 10

    monkeypatch.setenv("SORA_TELEGRAM_TOKEN", "env-token")
    monkeypatch.setenv("SORA_TELEGRAM_ALLOW_FROM", "22, 33")
    monkeypatch.setenv("SORA_TELEGRAM_PARSE_MODE", "Markdown")
    monkeypatch.setenv("SORA_TELEGRAM_POLL_TIMEOUT", "20")
    monkeypatch.setenv("SORA_TELEGRAM_DOWNLOAD_FILES", "off")
    monkeypatch.setenv("SORA_TELEGRAM_MAX_ATTACHMENT_BYTES", "12345")

    config = gateway._telegram_config(
        settings,
        {"SORA_TELEGRAM_TOKEN": "saved-token"},
        SimpleNamespace(token=None, allow=None),
    )

    assert config.token == "env-token"
    assert config.allow_from == ("22", "33")
    assert config.parse_mode == "Markdown"
    assert config.poll_timeout == 20
    assert config.download_attachments is False
    assert config.max_attachment_bytes == 12345


def test_explicit_flags_win_over_environment(monkeypatch):
    _clear_runtime_overrides(monkeypatch)
    settings = recommended_settings()
    monkeypatch.setenv("SORA_TELEGRAM_TOKEN", "env-token")
    monkeypatch.setenv("SORA_TELEGRAM_ALLOW_FROM", "22")

    config = gateway._telegram_config(
        settings,
        {},
        SimpleNamespace(token="flag-token", allow="44"),
    )

    assert config.token == "flag-token"
    assert config.allow_from == ("44",)


def test_telegram_config_reports_a_missing_token(monkeypatch):
    _clear_runtime_overrides(monkeypatch)

    with pytest.raises(ValueError, match="SORA_TELEGRAM_TOKEN"):
        gateway._telegram_config(
            recommended_settings(),
            {},
            SimpleNamespace(token=None, allow=None),
        )


# ---------------------------------------------------------------------------
# Entry point: configured and legacy behaviour
# ---------------------------------------------------------------------------


def test_gateway_uses_the_saved_configuration(monkeypatch, tmp_path):
    _clear_runtime_overrides(monkeypatch)
    settings = advanced_settings()
    settings.agent.name = "Nimbus"
    settings.provider.kind = "mock"
    settings.provider.model = "mock"
    settings.channel.allow_from = ("7",)
    save_settings(settings)
    monkeypatch.setenv("SORA_TELEGRAM_TOKEN", "token")
    captured = _capture_serve(monkeypatch)

    assert gateway.main([]) == 0

    kwargs = captured["kwargs"]
    assert captured["config"].allow_from == ("7",)
    assert kwargs["approval_mode"] == "off"
    assert kwargs["quality"] is False
    assert kwargs["clock"] is False
    assert kwargs["max_steps"] == 8
    assert kwargs["gateway_label"] == "☁️ sora-gateway"
    assert kwargs["file_understander"] is None
    assert captured["config"].download_attachments is False
    assert "Your name is Nimbus." in kwargs["system_prompt"]


def test_gateway_environment_still_overrides_saved_configuration(
    monkeypatch, tmp_path
):
    _clear_runtime_overrides(monkeypatch)
    settings = recommended_settings()
    settings.provider.kind = "mock"
    settings.provider.model = "mock"
    settings.channel.allow_from = ("7",)
    settings.features.memory.enabled = False
    settings.features.identity = False
    settings.paths.sessions_root = str(tmp_path / "sessions")
    save_settings(settings)
    monkeypatch.setenv("SORA_TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("SORA_APPROVAL", "off")
    monkeypatch.setenv("SORA_QUALITY", "0")
    monkeypatch.setenv("SORA_CLOCK", "0")
    captured = _capture_serve(monkeypatch)

    assert gateway.main([]) == 0

    kwargs = captured["kwargs"]
    assert kwargs["approval_mode"] == "off"
    assert kwargs["quality"] is False
    assert kwargs["clock"] is False
    assert kwargs["file_understander"] is not None
    assert captured["config"].download_attachments is True


def test_gateway_without_a_config_keeps_legacy_defaults(monkeypatch, tmp_path):
    _clear_runtime_overrides(monkeypatch)
    monkeypatch.setenv("SORA_TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("SORA_TELEGRAM_ALLOW_FROM", "7")
    monkeypatch.setenv("SORA_SESSIONS_ROOT", str(tmp_path / "sessions"))
    captured = _capture_serve(monkeypatch)

    assert gateway.main([]) == 0

    kwargs = captured["kwargs"]
    assert kwargs["approval_mode"] == "ask"
    assert kwargs["quality"] is True
    assert kwargs["clock"] is True
    assert kwargs["max_steps"] == 8
    assert kwargs["gateway_label"] == "sora-gateway"
    assert kwargs["file_understander"] is not None
    assert kwargs["context_window"] == legacy_settings().provider.context_window
    assert kwargs["system_prompt"].startswith(
        "You are a careful assistant talking with a user over a chat channel."
    )
    assert captured["channel_cls"].__name__ == "FakeChannel"


def test_gateway_reports_missing_channel_credentials(monkeypatch, capsys):
    _clear_runtime_overrides(monkeypatch)
    settings = advanced_settings()
    settings.channel.kind = "discord"
    save_settings(settings)

    assert gateway.main([]) == 2
    assert "SORA_DISCORD_TOKEN" in capsys.readouterr().err


def test_gateway_is_still_exported_as_a_console_entry_point():
    assert gateway_main is gateway.main
