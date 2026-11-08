"""Signal channel configuration (signal-cli HTTP daemon)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class SignalConfig:
    http_url: str
    account: str
    allow_from: tuple[str, ...] = ()
    request_timeout: float = 30.0
    reconnect_delay: float = 2.0
    max_message_chars: int = 4000
    download_attachments: bool = False
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True

    def __post_init__(self) -> None:
        if not self.http_url.strip():
            raise ValueError("SORA_SIGNAL_HTTP_URL must not be empty")
        if not self.account.strip():
            raise ValueError("SORA_SIGNAL_ACCOUNT must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("signal request_timeout must be > 0")
        if self.reconnect_delay < 0:
            raise ValueError("signal reconnect_delay must be >= 0")
        if self.max_message_chars <= 0:
            raise ValueError("signal max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("signal max_attachment_bytes must be > 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    http_url = env_str("SORA_SIGNAL_HTTP_URL")
    account = env_str("SORA_SIGNAL_ACCOUNT")
    if not http_url:
        raise ValueError(
            "SORA_SIGNAL_HTTP_URL is not set; start signal-cli with "
            "`signal-cli daemon --http 127.0.0.1:8080`"
        )
    if not account:
        raise ValueError(
            "SORA_SIGNAL_ACCOUNT is not set; use the bot's E.164 phone number"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_SIGNAL_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return SignalConfig(
        http_url=http_url.rstrip("/"),
        account=account,
        allow_from=allow_from,
        request_timeout=env_float("SORA_SIGNAL_REQUEST_TIMEOUT", 30.0),
        reconnect_delay=env_float("SORA_SIGNAL_RECONNECT_DELAY", 2.0),
        max_message_chars=env_int(
            "SORA_SIGNAL_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        download_attachments=env_bool("SORA_SIGNAL_DOWNLOAD_FILES", False),
        max_attachment_bytes=env_int(
            "SORA_SIGNAL_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
    )
