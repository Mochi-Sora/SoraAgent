"""Microsoft Teams Bot Framework transport built on the standard library."""

from __future__ import annotations

import threading
import time
import urllib.parse
from typing import Any, Protocol

from .._http import HttpError, download, request_json
from .._jwt import JwtError, fetch_jwks, verify_rs256


class TeamsError(RuntimeError):
    """A Teams or Bot Framework request failed."""


class TeamsTransport(Protocol):
    def access_token(self) -> str:
        """Return a cached Bot Framework access token."""
        ...

    def send_text(
        self,
        service_url: str,
        conversation_id: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        """Send one text activity."""
        ...

    def send_typing(self, service_url: str, conversation_id: str) -> None:
        """Send a typing activity."""
        ...

    def verify_jwt(self, token: str, service_url: str | None = None) -> dict[str, Any]:
        """Verify a Bot Framework bearer token."""
        ...

    def download(self, url: str, max_bytes: int) -> bytes:
        """Download one attachment."""
        ...


class UrllibTeamsTransport:
    """Bot Framework REST client with client-credentials authentication."""

    def __init__(
        self,
        app_id: str,
        app_password: str,
        *,
        token_url: str,
        openid_config_url: str,
        scope: str,
        request_timeout: float = 30.0,
    ) -> None:
        self._app_id = app_id
        self._app_password = app_password
        self._token_url = token_url
        self._openid_config_url = openid_config_url
        self._scope = scope
        self._request_timeout = float(request_timeout)
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()
        self._openid: dict[str, Any] | None = None
        self._openid_expires_at = 0.0
        self._openid_lock = threading.Lock()

    def access_token(self) -> str:
        now = time.time()
        with self._token_lock:
            if self._token and now < self._token_expires_at:
                return self._token
            payload = request_json(
                self._token_url,
                method="POST",
                form_body={
                    "grant_type": "client_credentials",
                    "client_id": self._app_id,
                    "client_secret": self._app_password,
                    "scope": self._scope,
                },
                timeout=self._request_timeout,
            )
            if not isinstance(payload, dict) or not payload.get("access_token"):
                raise TeamsError("teams token endpoint returned no access_token")
            self._token = str(payload["access_token"])
            expires_in = float(payload.get("expires_in") or 3600)
            self._token_expires_at = now + max(60.0, expires_in - 60.0)
            return self._token

    def send_text(
        self,
        service_url: str,
        conversation_id: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> str | None:
        activity: dict[str, Any] = {"type": "message", "text": text}
        if reply_to:
            activity["replyToId"] = str(reply_to)
        payload = self._send_activity(service_url, conversation_id, activity)
        if isinstance(payload, dict) and payload.get("id") is not None:
            return str(payload["id"])
        return None

    def send_typing(self, service_url: str, conversation_id: str) -> None:
        self._send_activity(service_url, conversation_id, {"type": "typing"})

    def verify_jwt(self, token: str, service_url: str | None = None) -> dict[str, Any]:
        config = self._openid_config()
        jwks_uri = config.get("jwks_uri")
        issuer = config.get("issuer")
        if not jwks_uri:
            raise TeamsError("Bot Framework OpenID configuration has no jwks_uri")
        jwks = fetch_jwks(str(jwks_uri), timeout=self._request_timeout)
        try:
            claims = verify_rs256(
                token,
                jwks,
                audience=self._app_id,
                issuer=None if issuer is None else str(issuer),
            )
        except JwtError as exc:
            raise TeamsError(f"invalid Teams token: {exc}") from exc
        if service_url:
            claimed = claims.get("serviceUrl")
            if not isinstance(claimed, str) or _normalize_url(
                claimed
            ) != _normalize_url(service_url):
                raise TeamsError("Teams token serviceUrl does not match the activity")
        return claims

    def download(self, url: str, max_bytes: int) -> bytes:
        try:
            return download(
                url,
                headers={"Authorization": f"Bearer {self.access_token()}"},
                timeout=self._request_timeout,
                max_bytes=max_bytes,
            )
        except HttpError as exc:
            raise TeamsError(str(exc)) from exc

    def _send_activity(
        self,
        service_url: str,
        conversation_id: str,
        activity: dict[str, Any],
    ) -> Any:
        base = service_url.rstrip("/")
        conversation = urllib.parse.quote(str(conversation_id), safe="")
        url = f"{base}/v3/conversations/{conversation}/activities"
        try:
            return request_json(
                url,
                method="POST",
                headers={"Authorization": f"Bearer {self.access_token()}"},
                json_body=activity,
                timeout=self._request_timeout,
                expected=(200, 201, 202),
            )
        except HttpError as exc:
            raise TeamsError(f"teams activity failed: {exc}") from exc

    def _openid_config(self) -> dict[str, Any]:
        now = time.time()
        with self._openid_lock:
            if self._openid is not None and now < self._openid_expires_at:
                return self._openid
            try:
                payload = request_json(
                    self._openid_config_url,
                    timeout=self._request_timeout,
                )
            except HttpError as exc:
                raise TeamsError(f"failed to read Bot Framework OpenID config: {exc}") from exc
            if not isinstance(payload, dict):
                raise TeamsError("Bot Framework OpenID config is not an object")
            self._openid = payload
            self._openid_expires_at = now + 3600.0
            return payload


def _normalize_url(value: str) -> str:
    return value.strip().rstrip("/").lower()
