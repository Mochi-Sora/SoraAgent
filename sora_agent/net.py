"""Small HTTPS helpers that work on python.org, Homebrew, and Linux builds."""

from __future__ import annotations

import importlib
import importlib.util
import os
import ssl
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    """Return a CA bundle that works even when Python has no default certs."""
    certifi = None
    if importlib.util.find_spec("certifi") is not None:
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


def urlopen(request: urllib.request.Request, timeout: float):
    """Open a request with the shared CA bundle for HTTPS URLs."""
    if request.full_url.lower().startswith("https://"):
        return urllib.request.urlopen(
            request, timeout=timeout, context=ssl_context()
        )
    return urllib.request.urlopen(request, timeout=timeout)
