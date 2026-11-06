"""Tests for the optional ``sora`` command line and its pure helpers."""

from __future__ import annotations

import io

import pytest
from typer.testing import CliRunner

from sora_agent.channels.types import (
    STREAM_DELTA_KIND,
    STREAM_END_KIND,
    OutboundMessage,
)
from sora_agent.cli.chat import ConsoleEmitter
from sora_agent.cli.main import app
from sora_agent.cli.setup import (
    _configure_files,
    _mask,
    _validate_allow,
    _validate_float,
    _validate_int,
    _validate_nonempty,
    run_wizard,
)
from sora_agent.settings import (
    advanced_settings,
    load_secrets,
    load_settings,
    recommended_settings,
    save_settings,
)

runner = CliRunner()


def _invoke(*args: str, input_text: str | None = None):
    return runner.invoke(app, list(args), input=input_text)


def _loaded():
    settings = load_settings()
    assert settings is not None
    return settings


# ---------------------------------------------------------------------------
# Root command
# ---------------------------------------------------------------------------


def test_version_flag():
    result = _invoke("--version")

    assert result.exit_code == 0
    assert "sora-agent" in result.stdout


def test_help_lists_the_primary_commands():
    result = _invoke("--help")

    assert result.exit_code == 0
    for name in ("setup", "chat", "run", "gateway", "doctor", "update", "config"):
        assert name in result.stdout


# ---------------------------------------------------------------------------
# config get/set/unset
# ---------------------------------------------------------------------------


def test_configure_files_reads_limits(monkeypatch):
    settings = recommended_settings()
    answers = iter([5, 1000, 2000])
    monkeypatch.setattr(
        "sora_agent.cli.setup.prompts.confirm", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "sora_agent.cli.setup.prompts.text",
        lambda *args, **kwargs: "vision-x",
    )
    monkeypatch.setattr(
        "sora_agent.cli.setup._ask_int", lambda *args, **kwargs: next(answers)
    )

    _configure_files(settings)

    files = settings.features.files
    assert files.enabled is True
    assert files.max_file_bytes == 5 * 1024 * 1024
    assert files.max_chars == 1000
    assert files.total_max_chars == 2000
    assert files.vision_model == "vision-x"


def test_configure_files_can_disable_understanding(monkeypatch):
    settings = recommended_settings()
    monkeypatch.setattr(
        "sora_agent.cli.setup.prompts.confirm", lambda *args, **kwargs: False
    )

    _configure_files(settings)

    assert settings.features.files.enabled is False


def test_config_get_set_and_unset_round_trip():
    save_settings(recommended_settings())

    result = _invoke("config", "get", "provider.model")
    assert result.exit_code == 0
    assert recommended_settings().provider.model in result.stdout

    result = _invoke("config", "set", "provider.temperature", "0.25")
    assert result.exit_code == 0
    assert _loaded().provider.temperature == 0.25

    result = _invoke("config", "set", "features.quality", "false")
    assert result.exit_code == 0
    assert _loaded().features.quality is False

    result = _invoke("config", "unset", "provider.temperature")
    assert result.exit_code == 0
    assert _loaded().provider.temperature == 0.7


def test_config_rejects_unknown_keys_and_bad_values():
    save_settings(advanced_settings())

    result = _invoke("config", "get", "provider.nope")
    assert result.exit_code == 1

    result = _invoke("config", "set", "provider.temperature", "not-a-number")
    assert result.exit_code == 1
    assert _loaded().provider.temperature == 0.1

    result = _invoke("config", "set", "features.quality", "maybe")
    assert result.exit_code == 1


def test_config_list_prints_valid_json():
    save_settings(recommended_settings())

    result = _invoke("config", "list")

    assert result.exit_code == 0
    assert '"mode": "recommended"' in result.stdout


# ---------------------------------------------------------------------------
# Secrets and doctor
# ---------------------------------------------------------------------------


def test_set_key_writes_a_private_secrets_file():
    result = _invoke("set-key", "provider", input_text="sk-test\n")

    assert result.exit_code == 0
    assert load_secrets()["OPENAI_API_KEY"] == "sk-test"


def test_doctor_reports_no_configuration():
    result = _invoke("doctor")

    assert result.exit_code == 1
    assert "No configuration" in result.stdout


def test_doctor_reports_a_configured_mock_install():
    settings = advanced_settings()
    settings.provider.kind = "mock"
    settings.provider.model = "mock"
    save_settings(settings)

    result = _invoke("doctor")

    assert result.exit_code == 0
    assert "advanced" in result.stdout
    assert "Mock (offline)" in result.stdout
    assert "(not required)" in result.stdout
    assert "Approval is OFF" in result.stdout
    assert "Quality evaluation is OFF" in result.stdout


# ---------------------------------------------------------------------------
# Wizard helpers
# ---------------------------------------------------------------------------


def test_wizard_rejects_unknown_modes_before_touching_the_terminal():
    with pytest.raises(ValueError):
        run_wizard("quickstart")


@pytest.mark.parametrize("value", ["", "*", "1", "1, 2, -100"])
def test_allow_list_accepts_numeric_ids(value):
    assert _validate_allow(value) is True


@pytest.mark.parametrize("value", ["@sora", "1,two", "abc"])
def test_allow_list_rejects_non_numeric_ids(value):
    assert isinstance(_validate_allow(value), str)


def test_numeric_validators_enforce_bounds():
    assert _validate_int("5", 1, 10) is True
    assert isinstance(_validate_int("0", 1, 10), str)
    assert isinstance(_validate_int("11", 1, 10), str)
    assert _validate_float("0.5", 0.0, 2.0) is True
    assert isinstance(_validate_float("2.5", 0.0, 2.0), str)
    assert _validate_nonempty("x") is True
    assert isinstance(_validate_nonempty("   "), str)


def test_mask_never_prints_a_whole_secret():
    assert _mask("short") == "•••••"
    masked = _mask("1234567890")
    assert "1234567890" not in masked
    assert masked.startswith("1234")
    assert masked.endswith("7890")


# ---------------------------------------------------------------------------
# Terminal streaming renderer
# ---------------------------------------------------------------------------


def _stream(text: str, kind: str, stream_id: str = "one") -> OutboundMessage:
    return OutboundMessage(
        "cli",
        "local",
        text,
        metadata={"kind": kind, "stream_id": stream_id},
    )


def test_console_emitter_prints_only_new_text_and_a_final_newline():
    out = io.StringIO()
    emitter = ConsoleEmitter(out)

    emitter(_stream("hel", STREAM_DELTA_KIND))
    emitter(_stream("hello", STREAM_DELTA_KIND))
    emitter(_stream("", STREAM_END_KIND))

    assert out.getvalue() == "hello\n"


def test_console_emitter_handles_a_replaced_snapshot():
    out = io.StringIO()
    emitter = ConsoleEmitter(out)

    emitter(_stream("first draft", STREAM_DELTA_KIND))
    emitter(_stream("final answer", STREAM_DELTA_KIND))

    assert out.getvalue() == "first draftfinal answer"
