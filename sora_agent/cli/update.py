"""Update an installed SoraAgent without re-running the installer.

Two installation layouts are supported:

* an editable or source checkout: pull the current branch (or check out an
  explicit tag) and reinstall in place;
* an installer-managed virtual environment: download the latest GitHub archive
  and install it into the running interpreter.

The updater never touches user data under ``~/.sora_agent`` and refuses to
modify a checkout that has uncommitted changes.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import __version__
from . import ui

DEFAULT_REPO = "Mochi-Sora/SoraAgent"
DEFAULT_TIMEOUT = 30.0
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


class UpdateError(RuntimeError):
    """Raised when an update cannot be completed safely."""


@dataclass(frozen=True)
class InstallInfo:
    version: str
    checkout: Path | None
    editable: bool

    @property
    def kind(self) -> str:
        if self.checkout is None:
            return "installer-managed environment"
        return "editable checkout" if self.editable else "source checkout"


@dataclass(frozen=True)
class Target:
    ref: str
    label: str
    release: bool = False


# ---------------------------------------------------------------------------
# Installed layout
# ---------------------------------------------------------------------------


def _distribution():
    try:
        return importlib.metadata.distribution("sora-agent")
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_version() -> str:
    dist = _distribution()
    if dist is None:
        return __version__
    return dist.version or __version__


def _url_to_path(url: str) -> Path | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"", "file"}:
        return None
    value = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
    return Path(value) if value else None


def _read_direct_url(dist) -> dict:
    try:
        raw = dist.read_text("direct_url.json")
    except (AttributeError, OSError):
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _detect_install() -> InstallInfo:
    dist = _distribution()
    version = (dist.version if dist is not None else __version__) or __version__

    if dist is not None:
        direct = _read_direct_url(dist)
        checkout = _url_to_path(str(direct.get("url") or ""))
        editable = bool((direct.get("dir_info") or {}).get("editable"))
        if (
            checkout is not None
            and (checkout / "pyproject.toml").is_file()
            and (checkout / ".git").exists()
        ):
            return InstallInfo(version, checkout, editable)

    # Running from a source tree without installed metadata (for example
    # ``python -m sora_agent.cli``) still counts as an editable checkout.
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
        return InstallInfo(version, candidate, True)

    return InstallInfo(version, None, False)


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _detected_extras() -> tuple[str, ...]:
    extras: list[str] = []
    if _module_available("openai"):
        extras.append("openai")
    if _module_available("zstandard") or sys.version_info >= (3, 14):
        extras.append("memory")
    # The running ``sora`` command itself requires the CLI extra.
    extras.append("cli")
    return tuple(extras)


def _parse_extras(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return _detected_extras()
    text = raw.strip()
    if not text or text.lower() == "none":
        return ()
    extras: list[str] = []
    for part in text.split(","):
        name = part.strip()
        if name and name not in extras:
            extras.append(name)
    return tuple(extras)


# ---------------------------------------------------------------------------
# GitHub targets
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Return a CA bundle that works on python.org, Homebrew, and Linux."""
    certifi = None
    if _module_available("certifi"):
        try:
            certifi = importlib.import_module("certifi")
        except ImportError:
            certifi = None
    if certifi is not None:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except (OSError, ssl.SSLError):
            pass

    context = ssl.create_default_context()
    if context.get_ca_certs():
        return context

    for candidate in (
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/local/etc/openssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
    ):
        if not os.path.isfile(candidate):
            continue
        try:
            bundle = ssl.create_default_context(cafile=candidate)
        except (OSError, ssl.SSLError):
            continue
        if bundle.get_ca_certs():
            return bundle
    return context


def _urlopen(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(
        request, timeout=timeout, context=_ssl_context()
    )


def _fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"sora-agent/{__version__}",
        },
    )
    with _urlopen(request, timeout) as response:
        return response.read()


def _latest_release(repo: str, timeout: float) -> str | None:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        data = json.loads(_fetch(url, timeout).decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, ValueError):
        return None
    tag = data.get("tag_name") if isinstance(data, dict) else None
    if not tag:
        return None
    return str(tag).strip() or None


def _resolve_target(
    version: str, repo: str, timeout: float = DEFAULT_TIMEOUT
) -> Target:
    value = (version or "latest").strip()
    folded = value.lower()
    if folded == "latest":
        tag = _latest_release(repo, timeout)
        if tag:
            return Target(f"refs/tags/{tag}", tag, release=True)
        return Target("refs/heads/main", "main (no release found)")
    if folded in {"main", "master"}:
        return Target(f"refs/heads/{folded}", folded)
    tag = value if value.lower().startswith("v") else f"v{value}"
    return Target(f"refs/tags/{tag}", tag, release=True)


def _normalize_version(value: str) -> str:
    return value.strip().lstrip("vV")


# ---------------------------------------------------------------------------
# Git checkouts
# ---------------------------------------------------------------------------


def _run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise UpdateError(
            "git is not installed, so this checkout cannot be updated"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            f"git {' '.join(args)} timed out; check the remote and try again"
        ) from exc


def _git_output(checkout: Path, *args: str) -> str:
    result = _run_git(checkout, *args)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise UpdateError(
            f"git {' '.join(args)} failed" + (f":\n{detail}" if detail else "")
        )
    return result.stdout.strip()


def _checkout_branch(checkout: Path) -> str:
    branch = _git_output(checkout, "rev-parse", "--abbrev-ref", "HEAD")
    if not branch or branch == "HEAD":
        raise UpdateError(
            "the checkout is in detached HEAD state; switch to a branch such "
            "as main before running `sora update`"
        )
    return branch


def _ensure_clean(checkout: Path) -> None:
    if _git_output(checkout, "status", "--porcelain"):
        raise UpdateError(
            f"{checkout} has uncommitted changes. Commit or stash them, then "
            "run `sora update` again."
        )


def _commits_behind(checkout: Path, branch: str) -> int:
    _git_output(checkout, "fetch", "--quiet", "origin")
    result = _run_git(checkout, "rev-list", "--count", f"HEAD..origin/{branch}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise UpdateError(
            f"could not compare with origin/{branch}"
            + (f":\n{detail}" if detail else "")
        )
    try:
        return int(result.stdout.strip() or "0")
    except ValueError as exc:
        raise UpdateError("git returned an invalid commit count") from exc


def _update_checkout(
    info: InstallInfo, version: str, extras: Sequence[str], repo: str
) -> None:
    checkout = info.checkout
    assert checkout is not None
    _ensure_clean(checkout)

    if version.strip().lower() in {"latest", "main", "master"}:
        branch = _checkout_branch(checkout)
        _git_output(checkout, "pull", "--ff-only", "origin", branch)
    else:
        target = _resolve_target(version, repo)
        _git_output(checkout, "fetch", "--quiet", "--tags", "origin")
        _git_output(checkout, "checkout", target.label)

    _install_source(checkout, extras, editable=info.editable)


def _check_checkout(info: InstallInfo, version: str, repo: str) -> tuple[bool, str]:
    checkout = info.checkout
    assert checkout is not None
    if version.strip().lower() not in {"latest", "main", "master"}:
        target = _resolve_target(version, repo)
        return True, f"{target.label} is requested explicitly."
    branch = _checkout_branch(checkout)
    behind = _commits_behind(checkout, branch)
    if behind == 0:
        return False, f"Already up to date with origin/{branch}."
    return True, f"{behind} commit(s) behind origin/{branch}."


# ---------------------------------------------------------------------------
# Archive updates
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, timeout: float) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": f"sora-agent/{__version__}"}
    )
    try:
        with _urlopen(request, timeout) as response:
            length = response.headers.get("Content-Length")
            try:
                expected = int(length) if length else 0
            except (TypeError, ValueError):
                expected = 0
            if expected > MAX_ARCHIVE_BYTES:
                raise UpdateError("the update archive is unexpectedly large")
            total = 0
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise UpdateError(
                            "the update archive is unexpectedly large"
                        )
                    handle.write(chunk)
    except UpdateError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise UpdateError(f"could not download {url}: {exc}") from exc


def _safe_path(root: Path, name: str) -> Path:
    base = root.resolve()
    target = (base / name).resolve()
    if target != base and base not in target.parents:
        raise UpdateError(f"archive contains an unsafe path: {name}")
    return target


def _extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                _safe_path(dest, member.name)
                if member.issym() or member.islnk():
                    raise UpdateError("archive contains an unsupported link")
                if not (member.isfile() or member.isdir()):
                    continue
                tar.extract(member, dest)
    except UpdateError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise UpdateError(f"could not extract the update archive: {exc}") from exc


def _extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_path(dest, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise UpdateError("archive contains an unsupported link")
                bundle.extract(member, dest)
    except UpdateError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateError(f"could not extract the update archive: {exc}") from exc


def _find_source_root(directory: Path) -> Path:
    if (directory / "pyproject.toml").is_file():
        return directory
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / "pyproject.toml").is_file():
            return child
    raise UpdateError(
        "the downloaded archive did not contain a SoraAgent source tree"
    )


def _pip_available() -> bool:
    return importlib.util.find_spec("pip") is not None


def _install_source(
    source: Path, extras: Sequence[str], *, editable: bool
) -> None:
    if not _pip_available():
        raise UpdateError(
            "pip is not available in this environment; run "
            "`python -m ensurepip --upgrade` and try again"
        )

    spec = "." + (f"[{','.join(extras)}]" if extras else "")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "--upgrade",
        "--no-cache-dir",
    ]
    if editable:
        command.extend(["-e", spec])
    else:
        command.extend(["--force-reinstall", spec])

    try:
        result = subprocess.run(
            command,
            cwd=str(source),
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except OSError as exc:
        raise UpdateError(f"could not start pip: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            "pip timed out while installing the update; try again"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        tail = "\n".join(detail.splitlines()[-15:])
        raise UpdateError(
            "pip could not install the update"
            + (f":\n{tail}" if tail else "")
        )


def _install_archive(
    repo: str, target: Target, extras: Sequence[str], timeout: float
) -> None:
    suffix = ".zip" if os.name == "nt" else ".tar.gz"
    url = f"https://github.com/{repo}/archive/{target.ref}{suffix}"
    with tempfile.TemporaryDirectory(prefix="sora-agent-update-") as raw_tmp:
        tmp = Path(raw_tmp)
        archive = tmp / f"source{suffix}"
        _download(url, archive, timeout)
        extracted = tmp / "src"
        if suffix == ".zip":
            _extract_zip(archive, extracted)
        else:
            _extract_tar(archive, extracted)
        _install_source(
            _find_source_root(extracted), extras, editable=False
        )


def _check_archive(info: InstallInfo, target: Target) -> tuple[bool, str]:
    if not target.release:
        return (
            False,
            "No release tags found; `sora update` will reinstall main.",
        )
    if _normalize_version(target.label) == _normalize_version(info.version):
        return False, f"Already up to date at {target.label}."
    return True, f"{target.label} is available (installed v{info.version})."


# ---------------------------------------------------------------------------
# Command entry point
# ---------------------------------------------------------------------------


def run_update(
    *,
    check: bool = False,
    version: str = "latest",
    extras: str | None = None,
    repo: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    repository = (repo or os.getenv("SORA_REPO", "") or DEFAULT_REPO).strip()
    info = _detect_install()
    selected_extras = _parse_extras(extras)

    if info.checkout is not None and version.strip().lower() in {
        "latest",
        "main",
        "master",
    }:
        branch = _checkout_branch(info.checkout)
        target = Target(f"refs/heads/{branch}", f"origin/{branch}")
    else:
        target = _resolve_target(version, repository, timeout)

    ui.banner("Sora", "☁️", info.version, "Update")
    ui.kv_table(
        "Update plan",
        [
            ("Install", info.kind),
            ("Location", str(info.checkout or sys.prefix)),
            ("Current", f"v{info.version}"),
            ("Target", target.label),
            ("Extras", ", ".join(selected_extras) or "none"),
        ],
    )

    if check:
        if info.checkout is not None:
            available, message = _check_checkout(info, version, repository)
        else:
            available, message = _check_archive(info, target)
        if available:
            ui.warning(f"Update available: {message}")
            return 1
        ui.success(message)
        return 0

    ui.info(f"Updating from {target.label}...")
    if info.checkout is not None:
        _update_checkout(info, version, selected_extras, repository)
    else:
        _install_archive(repository, target, selected_extras, timeout)

    updated_version = _package_version()
    if _normalize_version(updated_version) != _normalize_version(info.version):
        ui.success(f"Updated to v{updated_version}.")
    else:
        ui.success(f"Reinstalled v{updated_version} from {target.label}.")
    ui.info("Restart any running gateway or chat to load the new code.")
    return 0
