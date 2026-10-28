import pytest


@pytest.fixture(autouse=True)
def _disable_default_memory(monkeypatch):
    monkeypatch.setenv("SORA_MEMORY", "0")
    monkeypatch.setenv("SORA_IDENTITY", "0")
