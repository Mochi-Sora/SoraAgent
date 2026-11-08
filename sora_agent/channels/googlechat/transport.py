"""Google Chat transport: signed events in, incoming webhook out."""

from __future__ import annotations

from typing import Any, Protocol

from .._http import HttpError, request_json
from .._jwt import JwtError, fetch_jwks, verify_rs256


class GoogleChatError(RuntimeError):
    """A Google Chat request failed."""


class GoogleChatTransport(Protocol):
    def send_webhook(self, text: str, *, thread_name: str | None = None) -> None:
        """Post one message through the configured incoming webhook."""
        ...

    def verify_jwt(self, token: str) -> dict[str, Any]:
        """Verify a Google Chat event bearer token."""
        ...


class UrllibGoogleChatTransport:
    """Minimal Google Chat client using ``urllib``."""

    def __init__(
        self,
        webhook_url: str,
        *,
        audience: str,
        issuer: str,
        jwks_url: str,
        request_timeout: float = 30.0,
    ) -> None:
        self._webhook_url = webhook_url
        self._audience = audience
        self._issuer = issuer
        self._jwks_url = jwks_url
        self._request_timeout = float(request_timeout)

    def send_webhook(self, text: str, *, thread_name: str | None = None) -> None:
        body: dict[str, Any] = {"text": text}
        if thread_name:
            body["thread"] = {"name": thread_name}
        try:
            request_json(
                self._webhook_url,
                method="POST",
                json_body=body,
                timeout=self._request_timeout,
            )
        except HttpError as exc:
            raise GoogleChatError(f"google chat webhook failed: {exc}") from exc

    def verify_jwt(self, token: str) -> dict[str, Any]:
        try:
            jwks = fetch_jwks(self._jwks_url, timeout=self._request_timeout)
            return verify_rs256(
                token,
                jwks,
                audience=self._audience,
                issuer=self._issuer,
            )
        except JwtError as exc:
            raise GoogleChatError(f"invalid Google Chat token: {exc}") from exc
