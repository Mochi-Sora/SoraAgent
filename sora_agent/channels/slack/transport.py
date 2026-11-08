"""Slack Web API transport built on the standard library."""

from __future__ import annotations

import time
from typing import Any, Protocol

from .._http import HttpError, download, request_json


class SlackError(RuntimeError):
    """A Slack API request failed."""


class SlackTransport(Protocol):
    def open_socket(self) -> str:
        """Return the Socket Mode WebSocket URL."""
        ...

    def auth_test(self) -> dict[str, Any]:
        """Return the bot identity."""
        ...

    def post_message(
        self,
        channel: str,
        text: str,
        *,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post one message and return its timestamp."""
        ...

    def delete_message(self, channel: str, ts: str) -> None:
        """Delete a message created by this bot."""
        ...

    def set_status(self, channel: str, thread_ts: str, status: str) -> None:
        """Set the assistant thread status."""
        ...

    def download(self, url: str, max_bytes: int) -> bytes:
        """Download one private file."""
        ...


class UrllibSlackTransport:
    """Minimal Slack Web API client using ``urllib``."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        api_base: str = "https://slack.com/api",
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._api_base = api_base.rstrip("/")
        self._request_timeout = float(request_timeout)

    def open_socket(self) -> str:
        payload = self._call(
            "apps.connections.open",
            token=self._app_token,
            json_body={},
        )
        url = payload.get("url") if isinstance(payload, dict) else None
        if not url:
            raise SlackError("slack apps.connections.open returned no url")
        return str(url)

    def auth_test(self) -> dict[str, Any]:
        payload = self._call("auth.test", token=self._bot_token, json_body={})
        return payload if isinstance(payload, dict) else {}

    def post_message(
        self,
        channel: str,
        text: str,
        *,
        thread_ts: str | None = None,
    ) -> str | None:
        body: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        if thread_ts:
            body["thread_ts"] = thread_ts
        payload = self._call("chat.postMessage", token=self._bot_token, json_body=body)
        ts = payload.get("ts") if isinstance(payload, dict) else None
        return None if ts is None else str(ts)

    def delete_message(self, channel: str, ts: str) -> None:
        self._call(
            "chat.delete",
            token=self._bot_token,
            json_body={"channel": channel, "ts": ts},
        )

    def set_status(self, channel: str, thread_ts: str, status: str) -> None:
        self._call(
            "assistant.threads.setStatus",
            token=self._bot_token,
            json_body={
                "channel_id": channel,
                "thread_ts": thread_ts,
                "status": status,
            },
        )

    def download(self, url: str, max_bytes: int) -> bytes:
        try:
            return download(
                url,
                headers={"Authorization": f"Bearer {self._bot_token}"},
                timeout=self._request_timeout,
                max_bytes=max_bytes,
            )
        except HttpError as exc:
            raise SlackError(str(exc)) from exc

    def _call(
        self,
        method: str,
        *,
        token: str,
        json_body: Any,
    ) -> Any:
        url = f"{self._api_base}/{method}"
        for attempt in range(2):
            try:
                payload = request_json(
                    url,
                    method="POST",
                    headers={"Authorization": f"Bearer {token}"},
                    json_body=json_body,
                    timeout=self._request_timeout,
                )
            except HttpError as exc:
                if exc.status == 429 and attempt == 0:
                    time.sleep(min(5.0, exc.retry_after or 1.0))
                    continue
                raise SlackError(f"slack {method} failed: {exc}") from exc
            if not isinstance(payload, dict):
                raise SlackError(f"slack {method} returned an invalid response")
            if payload.get("ok") is not True:
                error = payload.get("error") or "unknown_error"
                if error == "ratelimited" and attempt == 0:
                    time.sleep(1.0)
                    continue
                raise SlackError(f"slack {method} failed: {error}")
            return payload
        raise SlackError(f"slack {method} failed after retry")
