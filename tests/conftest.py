import pytest


@pytest.fixture(autouse=True)
def _disable_default_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("SORA_MEMORY", "0")
    monkeypatch.setenv("SORA_MEMORY_DREAM_AUTO", "0")
    monkeypatch.setenv("SORA_IDENTITY", "0")
    # Memory-enabled tests must never reach the network to download the sidecar.
    monkeypatch.setenv("SORA_MEMORY_AUTO_DOWNLOAD", "0")
    # Keep the default native calendar out of the real home directory.
    monkeypatch.setenv("SORA_CALENDAR_DB", str(tmp_path / "calendar.db"))
    # Never read or write the developer's real ~/.sora_agent configuration.
    monkeypatch.setenv("SORA_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("SORA_SECRETS", str(tmp_path / "secrets.env"))
