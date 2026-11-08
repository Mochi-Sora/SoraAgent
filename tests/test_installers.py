"""Safety checks for the shell installer's destructive paths."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def _guard_function() -> str:
    text = (ROOT / "install.sh").read_text(encoding="utf-8")
    start = text.index("is_dangerous_dir()")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _is_dangerous(path: str, home: Path) -> bool:
    script = (
        "set -u\n"
        + _guard_function()
        + '\nif is_dangerous_dir "$1"; then exit 0; else exit 1; fi\n'
    )
    environment = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    result = subprocess.run(
        [BASH or "bash", "-c", script, "bash", path],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_installer_refuses_root_home_and_ancestors(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    assert _is_dangerous("/", home) is True
    assert _is_dangerous("/tmp", home) is True
    assert _is_dangerous(str(home), home) is True
    assert _is_dangerous(str(home) + "/..", home) is True
    assert _is_dangerous(str(home) + "/../etc", home) is True
    assert _is_dangerous("relative/path", home) is True
    assert _is_dangerous("./relative", home) is True


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_installer_accepts_a_normal_nested_install_root(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()

    assert _is_dangerous(str(home / ".local" / "share" / "sora-agent"), home) is False
    assert _is_dangerous("/opt/sora-agent", home) is False


def test_uninstall_is_guarded_by_install_markers() -> None:
    text = (ROOT / "install.sh").read_text(encoding="utf-8")

    assert "does not look like a SoraAgent install" in text
    assert "managed by sora-agent installer" in text
    assert "refusing to remove" in text


def test_installer_wrappers_quote_their_target() -> None:
    text = (ROOT / "install.sh").read_text(encoding="utf-8")

    assert "printf '%q'" in text
    assert "exec %s" in text


def test_windows_installer_has_equivalent_guards() -> None:
    text = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert "Assert-SafeDirectory" in text
    assert "does not look like a SoraAgent install" in text
    assert "managed by sora-agent installer" in text
    assert "must not be a drive root" in text
