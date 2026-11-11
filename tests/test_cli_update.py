"""Tests for the ``sora update`` self-updater."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from sora_agent.cli import update


class _FakeUI:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _FakeDistribution:
    def __init__(self, version: str, direct_url: dict | None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name: str):
        if name == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None


def _checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname='x'\n")
    (checkout / ".git").mkdir()
    return checkout


# ---------------------------------------------------------------------------
# Detection and target resolution
# ---------------------------------------------------------------------------


def test_detect_install_reads_editable_direct_url(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    dist = _FakeDistribution(
        "0.1.0",
        {"url": checkout.as_uri(), "dir_info": {"editable": True}},
    )
    monkeypatch.setattr(update, "_distribution", lambda: dist)

    info = update._detect_install()

    assert info.version == "0.1.0"
    assert info.checkout == checkout
    assert info.editable is True
    assert info.kind == "editable checkout"


def test_detect_install_reads_source_checkout(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    dist = _FakeDistribution(
        "0.1.0",
        {"url": checkout.as_uri(), "dir_info": {}},
    )
    monkeypatch.setattr(update, "_distribution", lambda: dist)

    info = update._detect_install()

    assert info.checkout == checkout
    assert info.editable is False
    assert info.kind == "source checkout"


def test_parse_extras_none_detected_and_deduplicated(monkeypatch):
    monkeypatch.setattr(update, "_detected_extras", lambda: ("openai", "cli"))

    assert update._parse_extras("none") == ()
    assert update._parse_extras("") == ()
    assert update._parse_extras(" openai, cli,,openai ") == ("openai", "cli")
    assert update._parse_extras(None) == ("openai", "cli")


def test_detected_extras_always_include_files_and_cli(monkeypatch):
    # A legacy install has no pypdf yet, so detection must not depend on it.
    monkeypatch.setattr(update, "_module_available", lambda name: False)

    extras = update._detected_extras()

    assert "files" in extras
    assert "cli" in extras
    assert "openai" not in extras


def test_resolve_target_latest_falls_back_to_main_without_a_release(
    monkeypatch,
):
    monkeypatch.setattr(update, "_latest_release", lambda repo, timeout: None)

    target = update._resolve_target("latest", "owner/repo", 5.0)

    assert target.ref == "refs/heads/main"
    assert target.label == "main (no release found)"
    assert target.release is False


def test_resolve_target_allows_explicit_main(monkeypatch):
    target = update._resolve_target("main", "owner/repo", 5.0)

    assert target.ref == "refs/heads/main"
    assert target.release is False


def test_resolve_target_pins_release_tags(monkeypatch):
    monkeypatch.setattr(
        update, "_latest_release", lambda repo, timeout: "v0.3.0"
    )

    latest = update._resolve_target("latest", "owner/repo", 5.0)
    pinned = update._resolve_target("0.2.0", "owner/repo", 5.0)

    assert latest.ref == "refs/tags/v0.3.0"
    assert latest.release is True
    assert pinned.ref == "refs/tags/v0.2.0"
    assert pinned.label == "v0.2.0"


def test_normalize_version():
    assert update._normalize_version("v0.2.0") == "0.2.0"
    assert update._normalize_version("0.2.0") == "0.2.0"


# ---------------------------------------------------------------------------
# Archive safety
# ---------------------------------------------------------------------------


def _tar_with_member(path: Path, name: str) -> None:
    with tarfile.open(path, "w:gz") as tar:
        payload = b"x"
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def _zip_with_member(path: Path, name: str) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(name, "x")


def test_tar_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _tar_with_member(archive, "../escape.txt")

    with pytest.raises(update.UpdateError, match="unsafe path"):
        update._extract_tar(archive, tmp_path / "out")


def test_zip_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    _zip_with_member(archive, "../escape.txt")

    with pytest.raises(update.UpdateError, match="unsafe path"):
        update._extract_zip(archive, tmp_path / "out")


def test_tar_extraction_accepts_normal_tree(tmp_path):
    archive = tmp_path / "good.tar.gz"
    _tar_with_member(archive, "SoraAgent-main/pyproject.toml")

    update._extract_tar(archive, tmp_path / "out")

    assert (tmp_path / "out" / "SoraAgent-main" / "pyproject.toml").is_file()
    assert update._find_source_root(tmp_path / "out").name == "SoraAgent-main"


# ---------------------------------------------------------------------------
# Git and pip behavior
# ---------------------------------------------------------------------------


def test_dirty_checkout_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        update, "_git_output", lambda checkout, *args: " M README.md"
    )

    with pytest.raises(update.UpdateError, match="uncommitted changes"):
        update._ensure_clean(tmp_path)


def test_checkout_update_pulls_branch_and_reinstalls(tmp_path, monkeypatch):
    calls: list[tuple[str, ...]] = []
    installed: list[tuple[Path, tuple[str, ...], bool]] = []

    def fake_git(checkout: Path, *args: str) -> str:
        calls.append(args)
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "main"
        return ""

    def fake_install(source, extras, *, editable):
        installed.append((source, tuple(extras), editable))

    monkeypatch.setattr(update, "_git_output", fake_git)
    monkeypatch.setattr(update, "_install_source", fake_install)

    info = update.InstallInfo("0.1.0", tmp_path, True)
    update._update_checkout(info, "main", ("cli",), "owner/repo")

    assert ("pull", "--ff-only", "origin", "main") in calls
    assert installed == [(tmp_path, ("cli",), True)]


def test_install_source_builds_editable_and_forced_commands(tmp_path, monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(update, "_pip_available", lambda: True)
    monkeypatch.setattr(update.subprocess, "run", fake_run)

    update._install_source(tmp_path, ("openai", "cli"), editable=True)
    update._install_source(tmp_path, ("cli",), editable=False)

    editable_command = calls[0][0]
    forced_command = calls[1][0]
    assert "-e" in editable_command
    assert ".[openai,cli]" in editable_command
    assert "--force-reinstall" in forced_command
    assert ".[cli]" in forced_command
    assert calls[0][1]["cwd"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Command-level flows
# ---------------------------------------------------------------------------


def test_run_update_check_reports_available(monkeypatch):
    monkeypatch.setattr(update, "ui", _FakeUI())
    monkeypatch.setattr(
        update,
        "_detect_install",
        lambda: update.InstallInfo("0.1.0", None, False),
    )
    monkeypatch.setattr(
        update,
        "_resolve_target",
        lambda *args, **kwargs: update.Target(
            "refs/tags/v0.2.0", "v0.2.0", True
        ),
    )

    assert update.run_update(check=True) == 1


def test_run_update_check_reports_current(monkeypatch):
    monkeypatch.setattr(update, "ui", _FakeUI())
    monkeypatch.setattr(
        update,
        "_detect_install",
        lambda: update.InstallInfo("0.1.0", None, False),
    )
    monkeypatch.setattr(
        update,
        "_resolve_target",
        lambda *args, **kwargs: update.Target(
            "refs/tags/v0.1.0", "v0.1.0", True
        ),
    )

    assert update.run_update(check=True) == 0


def test_run_update_archive_flow_installs_and_reports(monkeypatch):
    monkeypatch.setattr(update, "ui", _FakeUI())
    monkeypatch.setattr(
        update,
        "_detect_install",
        lambda: update.InstallInfo("0.1.0", None, False),
    )
    monkeypatch.setattr(
        update,
        "_resolve_target",
        lambda *args, **kwargs: update.Target(
            "refs/heads/main", "main (no release found)"
        ),
    )
    captured: dict[str, object] = {}

    def fake_install(repo, target, extras, timeout):
        captured.update(
            repo=repo, target=target, extras=tuple(extras), timeout=timeout
        )

    monkeypatch.setattr(update, "_install_archive", fake_install)
    monkeypatch.setattr(update, "_package_version", lambda: "0.2.0")

    assert update.run_update(version="main", repo="owner/repo") == 0
    assert captured["repo"] == "owner/repo"
    assert captured["extras"] == update._detected_extras()
