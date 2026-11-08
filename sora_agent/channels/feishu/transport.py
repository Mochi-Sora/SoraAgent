"""Feishu / Lark Open API transport built on the standard library."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from typing import Any, Protocol

from .._http import HttpError, download, request_json


class FeishuError(RuntimeError):
    """A Feishu/Lark API request failed."""


class FeishuTransport(Protocol):
    def send_text(self, chat_id: str, text: str) -> str | None:
        """Send one text message."""
        ...

    def add_reaction(self, message_id: str, emoji: str) -> str | None:
        """Add one emoji reaction."""
        ...

    def delete_reaction(self, message_id: str, reaction_id: str) -> None:
        """Remove one emoji reaction."""
        ...

    def download_resource(
        self,
        message_id: str,
        file_key: str,
        resource_type: str,
        max_bytes: int,
    ) -> bytes:
        """Download an image or file from a message."""
        ...


class UrllibFeishuTransport:
    """Minimal Feishu/Lark client using ``urllib``."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        api_base: str,
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._api_base = api_base.rstrip("/")
        self._request_timeout = float(request_timeout)
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()

    def send_text(self, chat_id: str, text: str) -> str | None:
        payload = self._call(
            "POST",
            "/open-apis/im/v1/messages",
            query={"receive_id_type": "chat_id"},
            json_body={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        message_id = data.get("message_id") if isinstance(data, dict) else None
        return None if message_id is None else str(message_id)

    def add_reaction(self, message_id: str, emoji: str) -> str | None:
        encoded = urllib.parse.quote(message_id, safe="")
        payload = self._call(
            "POST",
            f"/open-apis/im/v1/messages/{encoded}/reactions",
            json_body={"reaction_type": {"emoji_type": emoji}},
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        reaction_id = data.get("reaction_id") if isinstance(data, dict) else None
        return None if reaction_id is None else str(reaction_id)

    def delete_reaction(self, message_id: str, reaction_id: str) -> None:
        encoded_message = urllib.parse.quote(message_id, safe="")
        encoded_reaction = urllib.parse.quote(reaction_id, safe="")
        self._call(
            "DELETE",
            f"/open-apis/im/v1/messages/{encoded_message}/reactions/"
            f"{encoded_reaction}",
        )

    def download_resource(
        self,
        message_id: str,
        file_key: str,
        resource_type: str,
        max_bytes: int,
    ) -> bytes:
        encoded_message = urllib.parse.quote(message_id, safe="")
        encoded_key = urllib.parse.quote(file_key, safe="")
        url = (
            f"{self._api_base}/open-apis/im/v1/messages/{encoded_message}"
            f"/resources/{encoded_key}?"
            + urllib.parse.urlencode(
                {"type": "image" if resource_type == "image" else "file"}
            )
        )
        try:
            return download(
                url,
                headers={"Authorization": f"Bearer {self.access_token()}"},
                timeout=self._request_timeout,
                max_bytes=max_bytes,
            )
        except HttpError as exc:
            raise FeishuError(f"feishu resource download failed: {exc}") from exc

    def access_token(self) -> str:
        now = time.time()
        with self._token_lock:
            if self._token and now < self._token_expires_at:
                return self._token
            payload = request_json(
                f"{self._api_base}/open-apis/auth/v3/tenant_access_token/internal",
                method="POST",
                json_body={
                    "app_id": self._app_id,
                    "app_secret": self._app_secret,
                },
                timeout=self._request_timeout,
            )
            if not isinstance(payload, dict) or not payload.get(
                "tenant_access_token"
            ):
                message = (
                    payload.get("msg") if isinstance(payload, dict) else None
                )
                raise FeishuError(
                    f"feishu token request failed: {message or 'unknown error'}"
                )
            self._token = str(payload["tenant_access_token"])
            expires = float(payload.get("expire") or 7200)
            self._token_expires_at = now + max(60.0, expires - 60.0)
            return self._token

    def _call(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        url = f"{self._api_base}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        try:
            payload = request_json(
                url,
                method=method,
                headers={"Authorization": f"Bearer {self.access_token()}"},
                json_body=json_body,
                timeout=self._request_timeout,
            )
        except HttpError as exc:
            raise FeishuError(f"feishu {method} {path} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise FeishuError(f"feishu {method} {path} returned invalid JSON")
        code = payload.get("code")
        if code not in (0, None):
            raise FeishuError(
                f"feishu {method} {path} failed: "
                f"{payload.get('msg') or code}"
            )
        return payload
