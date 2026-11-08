"""Minimal JSON/HTTP helpers built on the standard library."""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from ..net import urlopen


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


def download(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    max_bytes: int = 20 * 1024 * 1024,
) -> bytes:
    """Download at most ``max_bytes`` from ``url``."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    chunks: list[bytes] = []
    total = 0
    try:
        with urlopen(req, timeout) as response:
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
