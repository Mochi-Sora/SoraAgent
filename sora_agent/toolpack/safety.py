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
        roots = tuple(getattr(config, "allowed_roots", lambda: (config.root,))())
        if not any(resolved == root or root in resolved.parents for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise ValueError(
                f"path is outside the configured workspace: {resolved} "
                f"(allowed roots: {allowed}); set SORA_TOOLPACK_ROOTS to add "
                "another directory"
            )
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


def assert_public_peer(response: Any, url: str = "") -> None:
    """Reject a response whose actual peer is private after DNS resolution.

    The pre-flight URL check can be defeated by DNS rebinding: the name can
    resolve publicly for the check and privately for the connection.  This
    validates the connected socket, not the name.
    """
    sock = None
    try:
        sock = response.fp.raw._sock  # type: ignore[attr-defined]
    except AttributeError:
        sock = None
    if sock is None:
        raise ValueError(
            "could not verify the connection peer for " + (url or "the response")
        )
    try:
        peer = sock.getpeername()
    except OSError as exc:
        raise ValueError(f"could not verify the connection peer: {exc}") from exc
    address = str(peer[0]) if isinstance(peer, tuple) and peer else ""
    if not address or _is_private_address(address):
        raise ValueError(
            f"refusing a private, loopback, or unresolvable connection peer: {address or url}"
        )


def untrusted(source: str, text: str) -> str:
    """Frame third-party content as data rather than instructions."""

    return wrap_untrusted(source, text)
