"""Shared HTTPS context for the few outbound requests memory makes."""

from __future__ import annotations

import os
import ssl


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError, ssl.SSLError):
        pass  # fall back to the system trust store
    for path in (
        "/etc/ssl/cert.pem",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/certs/ca-certificates.crt",
    ):
        if os.path.exists(path):
            try:
                return ssl.create_default_context(cafile=path)
            except (OSError, ssl.SSLError):
                continue
    return ssl.create_default_context()
