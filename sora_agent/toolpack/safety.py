"""Path and network safety helpers for the core tool pack."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..untrusted import wrap_untrusted


def resolve_path(config: Any, value: str, *, must_exist: bool = False) -> Path:
    """Resolve a workspace path and optionally enforce the workspace root."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("path must not be empty")
    if "\x00" in raw:
        raise ValueError("path contains a NUL byte")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config.root / candidate
    resolved = candidate.resolve(strict=False)
    if config.restrict_to_root:
        try:
            resolved.relative_to(config.root)
        except ValueError as exc:
            raise ValueError(
                f"path is outside the configured workspace: {resolved}"
            ) from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def _is_private_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def assert_public_url(url: str, allow_private: bool = False) -> str:
    """Validate an HTTP(S) URL and reject local/private destinations."""

    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http:// and https:// URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not supported")
    if allow_private:
        return parsed.geturl()
    host = parsed.hostname
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve host: {host}") from exc
    addresses = {str(info[4][0]) for info in infos}
    if not addresses or any(_is_private_address(item) for item in addresses):
        raise ValueError(f"refusing a private, loopback, or unresolvable URL: {url}")
    return parsed.geturl()


def untrusted(source: str, text: str) -> str:
    """Frame third-party content as data rather than instructions."""

    return wrap_untrusted(source, text)
