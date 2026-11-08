"""BlueBubbles (iMessage) transport built on the standard library."""

from __future__ import annotations

import urllib.parse
import uuid
from typing import Any, Protocol

from .._http import HttpError, request_json


class IMessageError(RuntimeError):
    """A BlueBubbles request failed."""


class IMessageTransport(Protocol):
    def send_text(self, chat_guid: str, text: str) -> str | None:
        """Send one message."""
        ...

    def send_typing(self, chat_guid: str) -> None:
        """Show the typing indicator."""
        ...


class UrllibIMessageTransport:
    """Minimal BlueBubbles client using ``urllib``."""

    def __init__(
        self,
        server_url: str,
        password: str,
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._password = password
        self._request_timeout = float(request_timeout)

    def send_text(self, chat_guid: str, text: str) -> str | None:
        payload = self._call(
            "/api/v1/message/text",
            json_body={
                "chatGuid": chat_guid,
                "tempGuid": uuid.uuid4().hex,
                "message": text,
                "method": "apple-script",
            },
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        guid = data.get("guid") if isinstance(data, dict) else None
        return None if guid is None else str(guid)

    def send_typing(self, chat_guid: str) -> None:
        encoded = urllib.parse.quote(chat_guid, safe="")
        self._call(
            f"/api/v1/chat/{encoded}/typing",
            json_body={"typing": True},
        )

    def download_attachment(self, guid: str, max_bytes: int) -> bytes:
        # BlueBubbles exposes attachment bytes through a POST download endpoint.
        encoded = urllib.parse.quote(guid, safe="")
        try:
            _status, _headers, body = request_json_raw(
                self._url(f"/api/v1/message/attachment/{encoded}/download"),
                method="POST",
                json_body={},
                timeout=self._request_timeout,
            )
        except HttpError as exc:
            raise IMessageError(f"imessage attachment download failed: {exc}") from exc
        if len(body) > max_bytes:
            raise IMessageError(
                f"attachment is {len(body)} bytes, above the {max_bytes} byte limit"
            )
        return body

    def _call(self, path: str, *, json_body: Any) -> Any:
        try:
            payload = request_json(
                self._url(path),
                method="POST",
                json_body=json_body,
                timeout=self._request_timeout,
            )
        except HttpError as exc:
            raise IMessageError(f"imessage {path} failed: {exc}") from exc
        if isinstance(payload, dict):
            status = payload.get("status")
            if isinstance(status, int) and status >= 400:
                raise IMessageError(
                    f"imessage {path} failed: {payload.get('message') or status}"
                )
        return payload

    def _url(self, path: str) -> str:
        separator = "&" if "?" in path else "?"
        return f"{self._server_url}{path}{separator}" + urllib.parse.urlencode(
            {"password": self._password}
        )


def request_json_raw(*args, **kwargs):
    """Backward-compatible raw response helper for attachment downloads."""
    from .._http import request

    return request(*args, **kwargs)
