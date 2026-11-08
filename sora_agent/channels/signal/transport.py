"""signal-cli HTTP/JSON-RPC transport built on the standard library."""

from __future__ import annotations

import logging
import threading
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable
from typing import Any, Protocol

from ...net import urlopen
from .._http import HttpError, request_json
from .._sse import iter_sse

log = logging.getLogger(__name__)


class SignalError(RuntimeError):
    """A signal-cli request failed."""


class SignalTransport(Protocol):
    def events(self) -> Iterable[str]:
        """Yield signal-cli event lines until the stream closes."""
        ...

    def send_text(self, account: str, recipient: str, text: str) -> None:
        """Send one text message."""
        ...

    def send_typing(self, account: str, recipient: str) -> None:
        """Send a typing indicator."""
        ...

    def close(self) -> None:
        """Close the event stream."""
        ...


class UrllibSignalTransport:
    """Minimal client for ``signal-cli daemon --http``."""

    def __init__(
        self,
        http_url: str,
        account: str,
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self._http_url = http_url.rstrip("/")
        self._account = account
        self._request_timeout = float(request_timeout)
        self._response = None
        self._lock = threading.Lock()

    def events(self):
        url = (
            f"{self._http_url}/api/v1/events?"
            + urllib.parse.urlencode({"account": self._account})
        )
        request = urllib.request.Request(
            url,
            headers={"Accept": "text/event-stream"},
        )
        response = urlopen(request, self._request_timeout)
        with self._lock:
            self._response = response
        try:
            for line in response:
                yield line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line)
        finally:
            with self._lock:
                self._response = None
            try:
                response.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("signal event stream close failed: %s", exc)

    def send_text(self, account: str, recipient: str, text: str) -> None:
        self._rpc(
            "send",
            {
                "account": account,
                "recipient": [recipient],
                "message": text,
            },
        )

    def send_typing(self, account: str, recipient: str) -> None:
        self._rpc(
            "sendTyping",
            {
                "account": account,
                "recipient": [recipient],
            },
        )

    def close(self) -> None:
        with self._lock:
            response = self._response
            self._response = None
        if response is not None:
            try:
                response.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("signal event stream close failed: %s", exc)

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        try:
            payload = request_json(
                f"{self._http_url}/api/v1/rpc",
                method="POST",
                json_body={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": method,
                    "params": params,
                },
                timeout=self._request_timeout,
            )
        except HttpError as exc:
            raise SignalError(f"signal {method} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise SignalError(f"signal {method} returned an invalid response")
        error = payload.get("error")
        if error:
            raise SignalError(f"signal {method} failed: {error}")
        return payload.get("result")


def parse_event_lines(lines) -> list[dict[str, Any]]:
    """Parse an iterable of SSE lines into event objects."""
    return list(iter_sse(lines))
