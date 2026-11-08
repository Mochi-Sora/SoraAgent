"""Slack channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class SlackConfig:
    bot_token: str
    app_token: str
    allow_from: tuple[str, ...] = ()
    api_base: str = "https://slack.com/api"
    request_timeout: float = 30.0
    max_message_chars: int = 4000
    typing_mode: str = "auto"
    ack_text: str = "⏳ Preparing a response…"
    download_attachments: bool = True
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0

    def __post_init__(self) -> None:
        if not self.bot_token.strip():
            raise ValueError("SORA_SLACK_BOT_TOKEN must not be empty")
        if not self.app_token.strip():
            raise ValueError("SORA_SLACK_APP_TOKEN must not be empty")
        if not self.api_base.strip():
            raise ValueError("slack api_base must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("slack request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("slack max_message_chars must be > 0")
        if self.typing_mode not in {"auto", "status", "message", "off"}:
            raise ValueError(
                "slack typing_mode must be auto, status, message, or off"
            )
        if self.max_attachment_bytes <= 0:
            raise ValueError("slack max_attachment_bytes must be > 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    bot_token = (
        env_str("SORA_SLACK_BOT_TOKEN")
        or str(secrets.get("SORA_SLACK_BOT_TOKEN", "")).strip()
    )
    app_token = (
        env_str("SORA_SLACK_APP_TOKEN")
        or str(secrets.get("SORA_SLACK_APP_TOKEN", "")).strip()
    )
    if not bot_token:
        raise ValueError(
            "SORA_SLACK_BOT_TOKEN is not set; create a Slack app and install it "
            "to your workspace"
        )
    if not app_token:
        raise ValueError(
            "SORA_SLACK_APP_TOKEN is not set; enable Socket Mode and create an "
            "app-level token with connections:write"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_SLACK_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    typing_mode = env_str(
        "SORA_SLACK_TYPING_MODE", str(options.get("typing_mode") or "auto")
    ).lower()
    return SlackConfig(
        bot_token=bot_token,
        app_token=app_token,
        allow_from=allow_from,
        api_base=env_str(
            "SORA_SLACK_API_BASE",
            str(options.get("api_base") or "https://slack.com/api"),
        ),
        request_timeout=env_float("SORA_SLACK_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_SLACK_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        typing_mode=typing_mode,
        ack_text=env_str(
            "SORA_SLACK_ACK_TEXT",
            str(options.get("ack_text") or "⏳ Preparing a response…"),
        ),
        download_attachments=env_bool("SORA_SLACK_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_SLACK_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
        reconnect_attempts=env_int("SORA_SLACK_RECONNECT_ATTEMPTS", 5),
        reconnect_delay=env_float("SORA_SLACK_RECONNECT_DELAY", 2.0),
    )
