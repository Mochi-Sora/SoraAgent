"""Matrix channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class MatrixConfig:
    homeserver: str
    access_token: str
    user_id: str = ""
    allow_from: tuple[str, ...] = ()
    sync_timeout: int = 30
    request_timeout: float = 40.0
    reconnect_delay: float = 2.0
    max_message_chars: int = 4000
    download_attachments: bool = True
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True

    def __post_init__(self) -> None:
        if not self.homeserver.strip():
            raise ValueError("SORA_MATRIX_HOMESERVER must not be empty")
        if not self.access_token.strip():
            raise ValueError("SORA_MATRIX_ACCESS_TOKEN must not be empty")
        if not self.homeserver.startswith(("http://", "https://")):
            raise ValueError("matrix homeserver must start with http:// or https://")
        if self.sync_timeout < 0:
            raise ValueError("matrix sync_timeout must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("matrix request_timeout must be > 0")
        if self.reconnect_delay < 0:
            raise ValueError("matrix reconnect_delay must be >= 0")
        if self.max_message_chars <= 0:
            raise ValueError("matrix max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("matrix max_attachment_bytes must be > 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    homeserver = env_str("SORA_MATRIX_HOMESERVER")
    access_token = env_str("SORA_MATRIX_ACCESS_TOKEN") or str(
        secrets.get("SORA_MATRIX_ACCESS_TOKEN", "")
    ).strip()
    if not homeserver:
        raise ValueError(
            "SORA_MATRIX_HOMESERVER is not set; use the base URL of your "
            "Matrix homeserver"
        )
    if not access_token:
        raise ValueError(
            "SORA_MATRIX_ACCESS_TOKEN is not set; create an access token for "
            "the bot account"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_MATRIX_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return MatrixConfig(
        homeserver=homeserver.rstrip("/"),
        access_token=access_token,
        user_id=env_str("SORA_MATRIX_USER_ID"),
        allow_from=allow_from,
        sync_timeout=env_int(
            "SORA_MATRIX_SYNC_TIMEOUT",
            int(options.get("sync_timeout") or 30),
        ),
        request_timeout=env_float("SORA_MATRIX_REQUEST_TIMEOUT", 40.0),
        reconnect_delay=env_float("SORA_MATRIX_RECONNECT_DELAY", 2.0),
        max_message_chars=env_int(
            "SORA_MATRIX_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        download_attachments=env_bool("SORA_MATRIX_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_MATRIX_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
    )
