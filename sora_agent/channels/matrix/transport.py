"""Matrix Client-Server API transport built on the standard library."""

from __future__ import annotations

import urllib.parse
import uuid
from typing import Any, Protocol

from .._http import HttpError, download, request_json


class MatrixError(RuntimeError):
    """A Matrix request failed."""


class MatrixTransport(Protocol):
    def whoami(self) -> str:
        """Return the bot's Matrix user id."""
        ...

    def sync(self, since: str | None, timeout: int) -> dict[str, Any]:
        """Long-poll ``/sync``."""
        ...

    def send_message(
        self,
        room_id: str,
        text: str,
        *,
        thread_root: str | None = None,
        reply_to: str | None = None,
        txn_id: str | None = None,
    ) -> str | None:
        """Send one text message."""
        ...

    def send_typing(self, room_id: str, user_id: str, timeout_ms: int = 30000) -> None:
        """Send a typing notification."""
        ...

    def download_media(self, mxc_url: str, max_bytes: int) -> bytes:
        """Download one ``mxc://`` media object."""
        ...


class UrllibMatrixTransport:
    """Minimal Matrix client using ``urllib``."""

    def __init__(
        self,
        homeserver: str,
        access_token: str,
        *,
        request_timeout: float = 40.0,
    ) -> None:
        self._homeserver = homeserver.rstrip("/")
        self._access_token = access_token
        self._request_timeout = float(request_timeout)

    def whoami(self) -> str:
        payload = self._call("GET", "/_matrix/client/v3/account/whoami")
        user_id = payload.get("user_id") if isinstance(payload, dict) else None
        if not user_id:
            raise MatrixError("matrix whoami returned no user_id")
        return str(user_id)

    def sync(self, since: str | None, timeout: int) -> dict[str, Any]:
        query: dict[str, Any] = {"timeout": max(0, int(timeout)) * 1000}
        if since:
            query["since"] = since
        url = (
            f"{self._homeserver}/_matrix/client/v3/sync?"
            + urllib.parse.urlencode(query)
        )
        payload = self._request(
            "GET",
            url,
            timeout=self._request_timeout + max(0, int(timeout)),
        )
        if not isinstance(payload, dict):
            raise MatrixError("matrix sync returned an invalid response")
        return payload

    def send_message(
        self,
        room_id: str,
        text: str,
        *,
        thread_root: str | None = None,
        reply_to: str | None = None,
        txn_id: str | None = None,
    ) -> str | None:
        room = urllib.parse.quote(room_id, safe="")
        transaction = txn_id or uuid.uuid4().hex
        body: dict[str, Any] = {"msgtype": "m.text", "body": text}
        if thread_root:
            body["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root,
            }
        elif reply_to:
            body["m.relates_to"] = {"m.in_reply_to": {"event_id": reply_to}}
        payload = self._call(
            "PUT",
            f"/_matrix/client/v3/rooms/{room}/send/m.room.message/{transaction}",
            json_body=body,
        )
        event_id = payload.get("event_id") if isinstance(payload, dict) else None
        return None if event_id is None else str(event_id)

    def send_typing(
        self,
        room_id: str,
        user_id: str,
        timeout_ms: int = 30000,
    ) -> None:
        room = urllib.parse.quote(room_id, safe="")
        user = urllib.parse.quote(user_id, safe="")
        self._call(
            "PUT",
            f"/_matrix/client/v3/rooms/{room}/typing/{user}",
            json_body={"typing": True, "timeout": max(1000, int(timeout_ms))},
        )

    def download_media(self, mxc_url: str, max_bytes: int) -> bytes:
        if not mxc_url.startswith("mxc://"):
            raise MatrixError("matrix attachment URL is not an mxc:// URL")
        server, _, media_id = mxc_url[len("mxc://") :].partition("/")
        if not server or not media_id:
            raise MatrixError("matrix attachment URL is malformed")
        quoted_server = urllib.parse.quote(server, safe="")
        quoted_media = urllib.parse.quote(media_id, safe="")
        headers = {"Authorization": f"Bearer {self._access_token}"}
        candidates = (
            (
                f"{self._homeserver}/_matrix/client/v1/media/download/"
                f"{quoted_server}/{quoted_media}"
            ),
            (
                f"{self._homeserver}/_matrix/media/v3/download/"
                f"{quoted_server}/{quoted_media}"
            ),
        )
        last_error: Exception | None = None
        for url in candidates:
            try:
                return download(
                    url,
                    headers=headers,
                    timeout=self._request_timeout,
                    max_bytes=max_bytes,
                )
            except HttpError as exc:
                last_error = exc
                if exc.status not in (404, 405):
                    break
        raise MatrixError(str(last_error or "matrix media download failed"))

    def _call(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
    ) -> Any:
        return self._request(
            method,
            f"{self._homeserver}{path}",
            json_body=json_body,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            return request_json(
                url,
                method=method,
                headers={"Authorization": f"Bearer {self._access_token}"},
                json_body=json_body,
                timeout=timeout or self._request_timeout,
            )
        except HttpError as exc:
            raise MatrixError(f"matrix {method} failed: {exc}") from exc
