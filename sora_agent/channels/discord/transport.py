"""Discord REST transport built on the standard library."""

from __future__ import annotations

import time
from typing import Any, Protocol

from .._http import HttpError, download, request_json


class DiscordError(RuntimeError):
    """A Discord API request failed."""


class DiscordTransport(Protocol):
    def gateway_url(self) -> str:
        """Return the gateway WebSocket URL."""
        ...

    def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        """Send one message and return its id when available."""
        ...

    def send_typing(self, channel_id: str) -> None:
        """Show Discord's typing indicator."""
        ...

    def download(self, url: str, max_bytes: int) -> bytes:
        """Download one attachment."""
        ...


class UrllibDiscordTransport:
    """Minimal Discord REST client using ``urllib``."""

    def __init__(
        self,
        token: str,
        api_base: str = "https://discord.com/api/v10",
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._request_timeout = float(request_timeout)

    def gateway_url(self) -> str:
        payload = self._request("GET", "/gateway/bot")
        if not isinstance(payload, dict) or not payload.get("url"):
            raise DiscordError("discord gateway response has no url")
        return str(payload["url"])

    def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        body: dict[str, Any] = {
            "content": text,
            "allowed_mentions": {"parse": []},
        }
        if reply_to:
            body["message_reference"] = {
                "message_id": str(reply_to),
                "fail_if_not_exists": False,
            }
        payload = self._request(
            "POST", f"/channels/{channel_id}/messages", json_body=body
        )
        if isinstance(payload, dict) and payload.get("id") is not None:
            return str(payload["id"])
        return None

    def send_typing(self, channel_id: str) -> None:
        self._request(
            "POST",
            f"/channels/{channel_id}/typing",
            expected=(204, 200),
        )

    def download(self, url: str, max_bytes: int) -> bytes:
        try:
            return download(url, timeout=self._request_timeout, max_bytes=max_bytes)
        except HttpError as exc:
            raise DiscordError(str(exc)) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        expected: tuple[int, ...] = (200,),
    ) -> Any:
        url = f"{self._api_base}{path}"
        headers = {
            "Authorization": f"Bot {self._token}",
            "User-Agent": "SoraAgent (https://github.com/Mochi-Sora/SoraAgent)",
        }
        for attempt in range(2):
            try:
                return request_json(
                    url,
                    method=method,
                    headers=headers,
                    json_body=json_body,
                    timeout=self._request_timeout,
                    expected=expected,
                )
            except HttpError as exc:
                if exc.status == 429 and attempt == 0:
                    time.sleep(min(5.0, exc.retry_after or 1.0))
                    continue
                detail = (
                    f"HTTP {exc.status}: {exc}" if exc.status else str(exc)
                )
                raise DiscordError(
                    f"discord {method} {path} failed: {detail}"
                ) from exc
        raise DiscordError(f"discord {method} {path} failed after retry")
