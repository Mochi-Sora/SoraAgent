"""The setup wizard stores search-provider choices without leaking keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from sora_agent.cli import setup as setup_mod
from sora_agent.settings import load_secrets, recommended_settings


def test_setup_web_search_defaults_to_duckduckgo(monkeypatch) -> None:
    settings = recommended_settings()
    monkeypatch.setattr(
        setup_mod.prompts, "select", lambda *args, **kwargs: "duckduckgo"
    )
    monkeypatch.setattr(
        setup_mod.prompts,
        "password",
        lambda *args, **kwargs: pytest.fail("DuckDuckGo must not ask for a key"),
    )

    setup_mod._configure_web_search(settings, {})

    assert settings.features.web_search.provider == "duckduckgo"


def test_setup_web_search_saves_tavily_key(monkeypatch, tmp_path: Path) -> None:
    secrets_file = tmp_path / "secrets.env"
    monkeypatch.setenv("SORA_SECRETS", str(secrets_file))
    settings = recommended_settings()
    monkeypatch.setattr(
        setup_mod.prompts, "select", lambda *args, **kwargs: "tavily"
    )
    monkeypatch.setattr(
        setup_mod.prompts,
        "password",
        lambda *args, **kwargs: "tvly-test-key",
    )

    setup_mod._configure_web_search(settings, {})

    assert settings.features.web_search.provider == "tavily"
    assert load_secrets(secrets_file)["TAVILY_API_KEY"] == "tvly-test-key"


def test_code_packages_are_opt_in(monkeypatch) -> None:
    settings = recommended_settings()
    monkeypatch.setattr(
        setup_mod.prompts, "select", lambda *args, **kwargs: "http"
    )
    monkeypatch.setattr(
        setup_mod.prompts, "confirm", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        setup_mod,
        "subprocess",
        type("Subprocess", (), {"run": lambda *a, **k: pytest.fail("no install")}),
    )

    setup_mod._configure_code_packages(settings)


def test_code_packages_install_when_confirmed(monkeypatch) -> None:
    settings = recommended_settings()
    monkeypatch.setattr(setup_mod, "_code_package_state", list)
    monkeypatch.setattr(
        setup_mod.prompts, "select", lambda *args, **kwargs: "both"
    )
    monkeypatch.setattr(
        setup_mod.prompts, "confirm", lambda *args, **kwargs: True
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(setup_mod.subprocess, "run", fake_run)

    setup_mod._configure_code_packages(settings)

    assert calls
    assert calls[0][-2:] == ["requests", "numpy"]
