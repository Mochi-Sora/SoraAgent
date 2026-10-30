import pytest


@pytest.fixture(autouse=True)
def _disable_default_memory(monkeypatch):
    monkeypatch.setenv("SORA_MEMORY", "0")
    monkeypatch.setenv("SORA_MEMORY_DREAM_AUTO", "0")
    monkeypatch.setenv("SORA_IDENTITY", "0")
    # Memory-enabled tests must never reach the network to download the sidecar.
    monkeypatch.setenv("SORA_MEMORY_AUTO_DOWNLOAD", "0")
