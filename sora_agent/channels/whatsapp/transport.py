"""WhatsApp Cloud API transport built on the standard library."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Protocol

from .._http import HttpError, download, request_json


class WhatsAppError(RuntimeError):
    """A WhatsApp Cloud API request failed."""


class WhatsAppTransport(Protocol):
    def send_text(
        self,
        phone_number_id: str,
        to: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        """Send one text message."""
        ...

    def mark_read(
        self,
        phone_number_id: str,
        message_id: str,
        *,
        typing: bool = True,
    ) -> None:
        """Mark a message as read, optionally showing the typing indicator."""
        ...

    def download_media(self, media_id: str, max_bytes: int) -> tuple[bytes, str]:
        """Download one media object and return ``(data, mime_type)``."""
        ...

    def verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify an ``X-Hub-Signature-256`` header."""
        ...


class UrllibWhatsAppTransport:
    """Minimal Graph API client for WhatsApp Cloud API."""

    def __init__(
        self,
        access_token: str,
        api_base: str = "https://graph.facebook.com/v21.0",
        *,
        app_secret: str = "",
        request_timeout: float = 30.0,
    ) -> None:
        self._access_token = access_token
        self._api_base = api_base.rstrip("/")
        self._app_secret = app_secret
        self._request_timeout = float(request_timeout)

    def send_text(
        self,
        phone_number_id: str,
        to: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
        if reply_to:
            body["context"] = {"message_id": reply_to}
        payload = self._call(phone_number_id, body)
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("id") is not None:
                return str(first["id"])
        return None

    def mark_read(
        self,
        phone_number_id: str,
        message_id: str,
        *,
        typing: bool = True,
    ) -> None:
        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        if typing:
            body["typing_indicator"] = {"type": "text"}
        self._call(phone_number_id, body)

    def download_media(self, media_id: str, max_bytes: int) -> tuple[bytes, str]:
        info = self._get(media_id)
        if not isinstance(info, dict) or not info.get("url"):
            raise WhatsAppError("whatsapp media lookup returned no url")
        size = _as_int(info.get("file_size"))
        if size is not None and size > max_bytes:
            raise WhatsAppError(
                f"media is {size} bytes, above the {max_bytes} byte limit"
            )
        try:
            data = download(
                str(info["url"]),
                headers=self._headers(),
                timeout=self._request_timeout,
                max_bytes=max_bytes,
            )
        except HttpError as exc:
            raise WhatsAppError(str(exc)) from exc
        return data, str(info.get("mime_type") or "")

    def verify_signature(self, body: bytes, signature: str) -> bool:
        if not self._app_secret:
            return False
        expected = "sha256=" + hmac.new(
            self._app_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature.strip())

    def _call(self, phone_number_id: str, body: dict[str, Any]) -> Any:
        try:
            return request_json(
                f"{self._api_base}/{phone_number_id}/messages",
                method="POST",
                headers=self._headers(),
                json_body=body,
                timeout=self._request_timeout,
                expected=(200, 201),
            )
        except HttpError as exc:
            raise WhatsAppError(f"whatsapp send failed: {exc}") from exc

    def _get(self, path: str) -> Any:
        try:
            return request_json(
                f"{self._api_base}/{path}",
                headers=self._headers(),
                timeout=self._request_timeout,
            )
        except HttpError as exc:
            raise WhatsAppError(f"whatsapp media lookup failed: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
