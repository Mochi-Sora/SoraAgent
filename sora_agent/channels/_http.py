"""Minimal JSON/HTTP helpers built on the standard library."""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from ..net import ssl_context, urlopen
from ..toolpack.safety import assert_public_peer, assert_public_url


class HttpError(RuntimeError):
    """An HTTP request failed or returned an unexpected status."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.retry_after = retry_after


def request(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    json_body: Any = None,
    form_body: Mapping[str, Any] | None = None,
    data: bytes | None = None,
    timeout: float = 30.0,
    expected: tuple[int, ...] = (200,),
) -> tuple[int, Mapping[str, str], bytes]:
    """Perform one HTTP request and return ``(status, headers, body)``."""
    request_headers = dict(headers or {})
    payload = data
    if json_body is not None:
        if payload is not None:
            raise ValueError("pass only one of json_body, form_body, or data")
        payload = json.dumps(json_body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form_body is not None:
        if payload is not None:
            raise ValueError("pass only one of json_body, form_body, or data")
        payload = urllib.parse.urlencode(form_body).encode("utf-8")
        request_headers.setdefault(
            "Content-Type", "application/x-www-form-urlencoded"
        )
    req = urllib.request.Request(
        url,
        data=payload,
        headers=request_headers,
        method=method.upper(),
    )
    try:
        with urlopen(req, timeout) as response:
            status = int(getattr(response, "status", 200))
            body = response.read()
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        retry_after = _retry_after(exc.headers)
        if exc.code in expected:
            return exc.code, dict(exc.headers.items()), detail.encode("utf-8")
        raise HttpError(
            f"HTTP {exc.code} from {url}: {detail[:300]}",
            status=exc.code,
            body=detail,
            retry_after=retry_after,
        ) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"request to {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpError(f"request to {url} timed out") from exc
    except OSError as exc:
        raise HttpError(f"request to {url} failed: {exc}") from exc
    if status not in expected:
        detail = body.decode("utf-8", "replace")
        raise HttpError(
            f"HTTP {status} from {url}: {detail[:300]}",
            status=status,
            body=detail,
            retry_after=_retry_after(response_headers),
        )
    return status, response_headers, body


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    json_body: Any = None,
    form_body: Mapping[str, Any] | None = None,
    data: bytes | None = None,
    timeout: float = 30.0,
    expected: tuple[int, ...] = (200,),
) -> Any:
    """Perform a request and decode its JSON body."""
    _status, _headers, body = request(
        url,
        method=method,
        headers=headers,
        json_body=json_body,
        form_body=form_body,
        data=data,
        timeout=timeout,
        expected=expected,
    )
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise HttpError(f"invalid JSON from {url}") from exc


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so an allowed URL cannot bounce to a private host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _host_allowed(host: str, allowed_hosts: Sequence[str]) -> bool:
    value = str(host or "").strip("[]").lower().rstrip(".")
    for suffix in allowed_hosts:
        candidate = str(suffix or "").strip().lower().lstrip(".").rstrip(".")
        if candidate and (value == candidate or value.endswith("." + candidate)):
            return True
    return False


def download(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    max_bytes: int = 20 * 1024 * 1024,
    allowed_hosts: Sequence[str] | None = None,
    validate_public: bool = False,
) -> bytes:
    """Download at most ``max_bytes`` from ``url``.

    ``allowed_hosts`` restricts token-bearing or platform-provided attachment
    URLs.  ``validate_public`` additionally rejects private/loopback
    destinations.  Redirects are never followed by this helper.
    """
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HttpError("attachment URL must be http(s) with a hostname")
    if parsed.username or parsed.password:
        raise HttpError("attachment URL must not contain credentials")
    if allowed_hosts and not _host_allowed(parsed.hostname, allowed_hosts):
        raise HttpError(
            f"attachment host is not allowed: {parsed.hostname}"
        )
    if validate_public:
        try:
            assert_public_url(url, allow_private=False)
        except ValueError as exc:
            raise HttpError(f"unsafe attachment URL: {exc}") from exc
    req = urllib.request.Request(url, headers=dict(headers or {}))
    chunks: list[bytes] = []
    total = 0
    try:
        opener = urllib.request.build_opener(
            _NoRedirectHandler(),
            urllib.request.HTTPSHandler(context=ssl_context()),
        )
        with opener.open(req, timeout=timeout) as response:
            if validate_public:
                assert_public_peer(response, url)
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HttpError(
                        f"download from {url} exceeds the {max_bytes} byte limit"
                    )
                chunks.append(chunk)
    except HttpError:
        raise
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        raise HttpError(
            f"HTTP {exc.code} downloading {url}: {detail[:300]}",
            status=exc.code,
            body=detail,
            retry_after=_retry_after(exc.headers),
        ) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"download from {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpError(f"download from {url} timed out") from exc
    except OSError as exc:
        raise HttpError(f"download from {url} failed: {exc}") from exc
    return b"".join(chunks)


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - diagnostics are best effort
        return ""
    finally:
        with contextlib.suppress(Exception):
            exc.close()


def _retry_after(headers: Any) -> float | None:
    try:
        raw = headers.get("Retry-After") if headers is not None else None
    except Exception:  # noqa: BLE001 - header containers vary
        raw = None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None
