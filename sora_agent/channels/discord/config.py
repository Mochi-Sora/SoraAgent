"""Discord channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class DiscordConfig:
    token: str
    allow_from: tuple[str, ...] = ()
    api_base: str = "https://discord.com/api/v10"
    gateway_url: str = ""
    intents: int = 37377
    request_timeout: float = 30.0
    max_message_chars: int = 2000
    download_attachments: bool = True
    max_attachment_bytes: int = 8 * 1024 * 1024
    typing: bool = True
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("SORA_DISCORD_TOKEN must not be empty")
        if not self.api_base.strip():
            raise ValueError("discord api_base must not be empty")
        if self.intents <= 0:
            raise ValueError("discord intents must be > 0")
        if self.request_timeout <= 0:
            raise ValueError("discord request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("discord max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("discord max_attachment_bytes must be > 0")
        if self.reconnect_attempts < 0:
            raise ValueError("discord reconnect_attempts must be >= 0")
        if self.reconnect_delay < 0:
            raise ValueError("discord reconnect_delay must be >= 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    token = (
        env_str("SORA_DISCORD_TOKEN")
        or str(secrets.get("SORA_DISCORD_TOKEN", "")).strip()
    )
    if not token:
        raise ValueError(
            "SORA_DISCORD_TOKEN is not set; create a bot in the Discord "
            "Developer Portal and export its token"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_DISCORD_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return DiscordConfig(
        token=token,
        allow_from=allow_from,
        api_base=env_str(
            "SORA_DISCORD_API_BASE",
            str(options.get("api_base") or "https://discord.com/api/v10"),
        ),
        gateway_url=env_str("SORA_DISCORD_GATEWAY_URL", ""),
        intents=env_int(
            "SORA_DISCORD_INTENTS", int(options.get("intents") or 37377)
        ),
        request_timeout=env_float("SORA_DISCORD_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_DISCORD_MAX_CHARS",
            int(options.get("max_message_chars") or 2000),
        ),
        download_attachments=env_bool("SORA_DISCORD_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_DISCORD_MAX_ATTACHMENT_BYTES", 8 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
        reconnect_attempts=env_int("SORA_DISCORD_RECONNECT_ATTEMPTS", 5),
        reconnect_delay=env_float("SORA_DISCORD_RECONNECT_DELAY", 2.0),
    )
