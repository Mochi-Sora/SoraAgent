"""Google Chat channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class GoogleChatConfig:
    webhook_url: str
    audience: str = ""
    allow_from: tuple[str, ...] = ()
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8080
    webhook_path: str = "/googlechat"
    jwks_url: str = (
        "https://www.googleapis.com/service_accounts/v1/jwk/"
        "chat@system.gserviceaccount.com"
    )
    issuer: str = "chat@system.gserviceaccount.com"
    request_timeout: float = 30.0
    max_message_chars: int = 4000
    ack_text: str = "⏳ Preparing a response…"
    download_attachments: bool = False
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True
    verify_jwt: bool = True

    def __post_init__(self) -> None:
        if not self.webhook_url.strip():
            raise ValueError(
                "SORA_GOOGLE_CHAT_WEBHOOK_URL is not set; create an incoming "
                "webhook for the Google Chat space"
            )
        if not self.webhook_url.startswith("https://"):
            raise ValueError("google chat webhook_url must use https://")
        if self.verify_jwt and not self.audience.strip():
            raise ValueError(
                "SORA_GOOGLE_CHAT_AUDIENCE is required to verify Google Chat "
                "requests; set it to the app's project number"
            )
        if not self.webhook_path.startswith("/"):
            raise ValueError("google chat webhook_path must start with /")
        if self.webhook_port < 0:
            raise ValueError("google chat webhook_port must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("google chat request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("google chat max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("google chat max_attachment_bytes must be > 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    webhook_url = env_str("SORA_GOOGLE_CHAT_WEBHOOK_URL")
    audience = env_str("SORA_GOOGLE_CHAT_AUDIENCE")
    allow_unverified = env_bool("SORA_GOOGLE_CHAT_ALLOW_UNVERIFIED", False)
    if not webhook_url:
        raise ValueError(
            "SORA_GOOGLE_CHAT_WEBHOOK_URL is not set; create an incoming "
            "webhook for the Google Chat space"
        )
    if not audience and not allow_unverified:
        raise ValueError(
            "SORA_GOOGLE_CHAT_AUDIENCE is not set; use the Google Chat app's "
            "project number, or explicitly allow unverified requests"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_GOOGLE_CHAT_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return GoogleChatConfig(
        webhook_url=webhook_url,
        audience=audience,
        allow_from=allow_from,
        webhook_host=env_str(
            "SORA_GOOGLE_CHAT_WEBHOOK_HOST",
            str(options.get("webhook_host") or "127.0.0.1"),
        ),
        webhook_port=env_int(
            "SORA_GOOGLE_CHAT_WEBHOOK_PORT",
            int(options.get("webhook_port") or 8080),
        ),
        webhook_path=env_str(
            "SORA_GOOGLE_CHAT_WEBHOOK_PATH",
            str(options.get("webhook_path") or "/googlechat"),
        ),
        jwks_url=env_str(
            "SORA_GOOGLE_CHAT_JWKS_URL",
            str(
                options.get("jwks_url")
                or (
                    "https://www.googleapis.com/service_accounts/v1/jwk/"
                    "chat@system.gserviceaccount.com"
                )
            ),
        ),
        issuer=env_str(
            "SORA_GOOGLE_CHAT_ISSUER",
            str(options.get("issuer") or "chat@system.gserviceaccount.com"),
        ),
        request_timeout=env_float("SORA_GOOGLE_CHAT_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_GOOGLE_CHAT_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        ack_text=env_str(
            "SORA_GOOGLE_CHAT_ACK_TEXT",
            str(options.get("ack_text") or "⏳ Preparing a response…"),
        ),
        download_attachments=env_bool("SORA_GOOGLE_CHAT_DOWNLOAD_FILES", False),
        max_attachment_bytes=env_int(
            "SORA_GOOGLE_CHAT_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
        verify_jwt=not allow_unverified,
    )
