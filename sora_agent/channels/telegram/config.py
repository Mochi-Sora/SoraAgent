"""Telegram channel configuration, readable from the environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .._util import env_bool, env_float, env_int, env_str

_PARSE_MODES = frozenset({"", "HTML", "Markdown", "MarkdownV2"})


@dataclass(frozen=True)
class TelegramConfig:
    """Settings for one Telegram bot.

    ``allow_from`` is a list of numeric user ids, or ``"*"`` to allow anyone.
    The empty default denies every sender: a token is not an access policy.
    """

    token: str
    allow_from: tuple[str, ...] = ()
    api_base: str = "https://api.telegram.org"
    poll_timeout: int = 30
    request_timeout: float = 40.0
    parse_mode: str | None = None
    max_message_chars: int = 4096
    download_attachments: bool = True
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("telegram token must not be empty")
        if self.poll_timeout < 0:
            raise ValueError("telegram poll_timeout must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("telegram request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("telegram max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("telegram max_attachment_bytes must be > 0")
        if self.parse_mode is not None and self.parse_mode not in _PARSE_MODES:
            raise ValueError(
                "telegram parse_mode must be one of HTML, Markdown, MarkdownV2"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TelegramConfig:
        """Build a config from ``SORA_TELEGRAM_*`` variables."""
        source = os.environ if env is None else env
        token = source.get("SORA_TELEGRAM_TOKEN", "").strip()
        if not token:
            raise ValueError(
                "SORA_TELEGRAM_TOKEN is not set; create a bot with @BotFather "
                "and export the token it gives you"
            )
        parse_mode = source.get("SORA_TELEGRAM_PARSE_MODE", "").strip() or None
        return cls(
            token=token,
            allow_from=split_ids(source.get("SORA_TELEGRAM_ALLOW_FROM", "")),
            api_base=source.get(
                "SORA_TELEGRAM_API_BASE", "https://api.telegram.org"
            ).strip(),
            poll_timeout=_as_int(source.get("SORA_TELEGRAM_POLL_TIMEOUT"), 30),
            request_timeout=_as_float(
                source.get("SORA_TELEGRAM_REQUEST_TIMEOUT"), 40.0
            ),
            parse_mode=parse_mode,
            max_message_chars=_as_int(
                source.get("SORA_TELEGRAM_MAX_CHARS"), 4096
            ),
            download_attachments=_as_bool(
                source.get("SORA_TELEGRAM_DOWNLOAD_FILES"), True
            ),
            max_attachment_bytes=_as_int(
                source.get("SORA_TELEGRAM_MAX_ATTACHMENT_BYTES"),
                20 * 1024 * 1024,
            ),
        )


def split_ids(raw: str) -> tuple[str, ...]:
    """Split a comma-separated id list, dropping blanks."""
    from .._util import split_ids as _split_ids

    return _split_ids(raw)


def from_settings(settings, secrets, args):
    """Build a config from settings, secrets, the environment, and flags."""
    token = (
        getattr(args, "token", None)
        or env_str("SORA_TELEGRAM_TOKEN")
        or str(secrets.get("SORA_TELEGRAM_TOKEN", "")).strip()
    )
    if not token:
        raise ValueError(
            "SORA_TELEGRAM_TOKEN is not set; create a bot with @BotFather and "
            "run `sora setup`, export the token, or pass --token"
        )

    allow_flag = getattr(args, "allow", None)
    allow_env = os.getenv("SORA_TELEGRAM_ALLOW_FROM")
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    elif allow_env is not None:
        allow_from = split_ids(allow_env)
    else:
        allow_from = tuple(settings.channel.allow_from)

    parse_env = os.getenv("SORA_TELEGRAM_PARSE_MODE")
    parse_mode = (
        (parse_env.strip() or None)
        if parse_env is not None
        else settings.channel.parse_mode
    )

    return TelegramConfig(
        token=token,
        allow_from=allow_from,
        api_base=env_str("SORA_TELEGRAM_API_BASE", settings.channel.api_base),
        poll_timeout=env_int(
            "SORA_TELEGRAM_POLL_TIMEOUT", settings.channel.poll_timeout
        ),
        request_timeout=env_float(
            "SORA_TELEGRAM_REQUEST_TIMEOUT", settings.channel.request_timeout
        ),
        parse_mode=parse_mode,
        max_message_chars=env_int(
            "SORA_TELEGRAM_MAX_CHARS", settings.channel.max_message_chars
        ),
        download_attachments=env_bool("SORA_TELEGRAM_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_TELEGRAM_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
    )


def _as_int(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(raw: str | None, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
