from __future__ import annotations

import pytest

from sora_agent.providers import (
    OpenAIChatProvider,
    ProviderConfigError,
    default_provider,
)


def test_deepseek_never_borrows_the_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="DEEPSEEK_API_KEY"):
        default_provider("deepseek")


def test_qwen_never_borrows_the_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="DASHSCOPE_API_KEY"):
        default_provider("qwen")


def test_custom_provider_does_not_implicitly_use_the_openai_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

    provider = default_provider(
        "custom", base_url="https://example.test/v1"
    )

    assert getattr(provider, "api_key") == ""


def test_missing_openai_key_is_a_sentinel_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        default_provider("openai")


def test_explicit_keys_are_used_and_preserved(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    provider = default_provider("deepseek", api_key="explicit-key")

    assert getattr(provider, "api_key") == "explicit-key"


def test_direct_construction_requires_a_key_by_default() -> None:
    with pytest.raises(ProviderConfigError):
        OpenAIChatProvider()


def test_plain_http_remote_endpoint_is_rejected() -> None:
    with pytest.raises(ProviderConfigError, match="plain HTTP"):
        OpenAIChatProvider(api_key="k", base_url="http://example.test/v1")


def test_plain_http_loopback_is_allowed_for_local_servers() -> None:
    provider = OpenAIChatProvider(
        api_key="",
        base_url="http://127.0.0.1:11434/v1",
        require_api_key=False,
    )

    assert provider.base_url == "http://127.0.0.1:11434/v1"


def test_invalid_scheme_is_rejected() -> None:
    with pytest.raises(ProviderConfigError, match="must start with"):
        OpenAIChatProvider(api_key="k", base_url="example.test/v1")


def test_provider_error_is_a_runtime_error() -> None:
    assert issubclass(ProviderConfigError, RuntimeError)
