"""Keep the package version and project metadata in lockstep."""

from __future__ import annotations

import re
from pathlib import Path

from sora_agent import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_and_package_versions_match() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
