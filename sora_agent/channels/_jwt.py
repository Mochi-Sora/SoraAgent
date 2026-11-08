"""Zero-dependency RS256 JWT verification for webhook channel security.

Teams and Google Chat sign their webhook requests with RS256 JWTs.  This module
verifies those tokens against a JWKS document; it deliberately implements only
RS256 signature verification and the standard time/audience/issuer claims.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
import urllib.request
from collections.abc import Mapping
from typing import Any

from ..net import urlopen

_SHA256_DIGEST_INFO = bytes.fromhex(
    "3031300d060960864801650304020105000420"
)
_JWKS_CACHE: dict[str, tuple[float, Mapping[str, Any]]] = {}
_JWKS_LOCK = threading.Lock()


class JwtError(RuntimeError):
    """A JWT was malformed or failed verification."""


def b64url_decode(value: str) -> bytes:
    raw = value.encode("ascii")
    padding = b"=" * (-len(raw) % 4)
    try:
        return base64.urlsafe_b64decode(raw + padding)
    except (ValueError, TypeError) as exc:
        raise JwtError("invalid base64url value") from exc


def b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def decode_jwt(token: str) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    """Return ``(header, claims, signing_input, signature)``."""
    parts = token.split(".")
    if len(parts) != 3:
        raise JwtError("JWT must have three segments")
    header_raw, claims_raw, signature_raw = parts
    try:
        header = json.loads(b64url_decode(header_raw).decode("utf-8"))
        claims = json.loads(b64url_decode(claims_raw).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise JwtError("JWT contains invalid JSON") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise JwtError("JWT header and claims must be objects")
    signing_input = f"{header_raw}.{claims_raw}".encode("ascii")
    signature = b64url_decode(signature_raw)
    return header, claims, signing_input, signature


def verify_rs256(
    token: str,
    jwks: Mapping[str, Any],
    *,
    audience: str | None = None,
    issuer: str | None = None,
    now: float | None = None,
    leeway: float = 60.0,
) -> dict[str, Any]:
    """Verify a compact RS256 JWT and return its claims."""
    header, claims, signing_input, signature = decode_jwt(token)
    if header.get("alg") != "RS256":
        raise JwtError("JWT algorithm is not RS256")
    key = _select_key(jwks, header.get("kid"))
    n = int.from_bytes(b64url_decode(str(key["n"])), "big")
    e = int.from_bytes(b64url_decode(str(key["e"])), "big")
    _verify_signature(n, e, signing_input, signature)
    _validate_claims(
        claims,
        audience=audience,
        issuer=issuer,
        now=time.time() if now is None else now,
        leeway=leeway,
    )
    return claims


def fetch_jwks(
    url: str,
    *,
    timeout: float = 30.0,
    cache_ttl: float = 3600.0,
) -> Mapping[str, Any]:
    """Fetch and cache a JWKS document."""
    now = time.time()
    with _JWKS_LOCK:
        cached = _JWKS_CACHE.get(url)
        if cached is not None and cached[0] > now:
            return cached[1]
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout) as response:
            raw = response.read()
    except Exception as exc:  # surface one stable error type
        raise JwtError(f"failed to fetch JWKS from {url}: {exc}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise JwtError(f"JWKS from {url} is not valid JSON") from exc
    if not isinstance(document, dict) or not isinstance(
        document.get("keys"), list
    ):
        raise JwtError(f"JWKS from {url} has no keys")
    with _JWKS_LOCK:
        _JWKS_CACHE[url] = (now + max(1.0, cache_ttl), document)
    return document


def clear_jwks_cache() -> None:
    with _JWKS_LOCK:
        _JWKS_CACHE.clear()


def _select_key(jwks: Mapping[str, Any], kid: Any) -> Mapping[str, Any]:
    keys = jwks.get("keys")
    if not isinstance(keys, list) or not keys:
        raise JwtError("JWKS has no keys")
    candidates = [key for key in keys if isinstance(key, dict)]
    if kid is not None:
        for key in candidates:
            if key.get("kid") == kid:
                return key
        raise JwtError(f"no JWKS key matches kid {kid!r}")
    if len(candidates) == 1:
        return candidates[0]
    raise JwtError("JWT has no kid and the JWKS has multiple keys")


def _verify_signature(
    n: int,
    e: int,
    signing_input: bytes,
    signature: bytes,
) -> None:
    key_size = (n.bit_length() + 7) // 8
    if len(signature) != key_size:
        raise JwtError("JWT signature length does not match the RSA key")
    digest = hashlib.sha256(signing_input).digest()
    digest_info = _SHA256_DIGEST_INFO + digest
    padding_length = key_size - len(digest_info) - 3
    if padding_length < 8:
        raise JwtError("RSA key is too small for RS256")
    expected = (
        b"\x00\x01"
        + b"\xff" * padding_length
        + b"\x00"
        + digest_info
    )
    message = pow(int.from_bytes(signature, "big"), e, n)
    actual = message.to_bytes(key_size, "big")
    if not hmac.compare_digest(actual, expected):
        raise JwtError("JWT signature verification failed")


def _validate_claims(
    claims: Mapping[str, Any],
    *,
    audience: str | None,
    issuer: str | None,
    now: float,
    leeway: float,
) -> None:
    exp = claims.get("exp")
    if exp is not None:
        try:
            if now > float(exp) + leeway:
                raise JwtError("JWT has expired")
        except (TypeError, ValueError) as exc:
            raise JwtError("JWT exp claim is not a number") from exc
    nbf = claims.get("nbf")
    if nbf is not None:
        try:
            if now + leeway < float(nbf):
                raise JwtError("JWT is not valid yet")
        except (TypeError, ValueError) as exc:
            raise JwtError("JWT nbf claim is not a number") from exc
    if audience is not None:
        audiences = claims.get("aud")
        if isinstance(audiences, str):
            audiences = [audiences]
        if not isinstance(audiences, list) or audience not in audiences:
            raise JwtError("JWT audience does not match")
    if issuer is not None:
        issuers = claims.get("iss")
        if isinstance(issuers, str):
            issuers = [issuers]
        if not isinstance(issuers, list) or issuer not in issuers:
            raise JwtError("JWT issuer does not match")
