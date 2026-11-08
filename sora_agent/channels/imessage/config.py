"""iMessage (BlueBubbles) channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class IMessageConfig:
    server_url: str
    password: str
    allow_from: tuple[str, ...] = ()
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8080
    webhook_path: str = "/imessage"
    webhook_secret: str = ""
    request_timeout: float = 30.0
    max_message_chars: int = 4000
    download_attachments: bool = False
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True

    def __post_init__(self) -> None:
        if not self.server_url.strip():
            raise ValueError(
                "SORA_IMESSAGE_SERVER_URL is not set; use the BlueBubbles "
                "server URL, for example http://127.0.0.1:1234"
            )
        if not self.server_url.startswith(("http://", "https://")):
            raise ValueError("imessage server_url must start with http:// or https://")
        if not self.password.strip():
            raise ValueError("SORA_IMESSAGE_PASSWORD must not be empty")
        if not self.webhook_path.startswith("/"):
            raise ValueError("imessage webhook_path must start with /")
        if self.webhook_port < 0:
            raise ValueError("imessage webhook_port must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("imessage request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("imessage max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("imessage max_attachment_bytes must be > 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    server_url = env_str("SORA_IMESSAGE_SERVER_URL")
    password = env_str("SORA_IMESSAGE_PASSWORD") or str(
        secrets.get("SORA_IMESSAGE_PASSWORD", "")
    ).strip()
    if not server_url:
        raise ValueError(
            "SORA_IMESSAGE_SERVER_URL is not set; point it at your "
            "BlueBubbles server"
        )
    if not password:
        raise ValueError(
            "SORA_IMESSAGE_PASSWORD is not set; use the BlueBubbles server "
            "password"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_IMESSAGE_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return IMessageConfig(
        server_url=server_url.rstrip("/"),
        password=password,
        allow_from=allow_from,
        webhook_host=env_str(
            "SORA_IMESSAGE_WEBHOOK_HOST",
            str(options.get("webhook_host") or "127.0.0.1"),
        ),
        webhook_port=env_int(
            "SORA_IMESSAGE_WEBHOOK_PORT",
            int(options.get("webhook_port") or 8080),
        ),
        webhook_path=env_str(
            "SORA_IMESSAGE_WEBHOOK_PATH",
            str(options.get("webhook_path") or "/imessage"),
        ),
        webhook_secret=env_str(
            "SORA_IMESSAGE_WEBHOOK_SECRET",
            str(options.get("webhook_secret") or ""),
        ),
        request_timeout=env_float("SORA_IMESSAGE_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_IMESSAGE_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        download_attachments=env_bool("SORA_IMESSAGE_DOWNLOAD_FILES", False),
        max_attachment_bytes=env_int(
            "SORA_IMESSAGE_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
    )
