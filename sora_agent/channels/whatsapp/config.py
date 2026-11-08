"""WhatsApp Cloud API channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class WhatsAppConfig:
    phone_number_id: str
    access_token: str
    verify_token: str
    app_secret: str = ""
    allow_from: tuple[str, ...] = ()
    api_base: str = "https://graph.facebook.com/v21.0"
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8080
    webhook_path: str = "/whatsapp"
    request_timeout: float = 30.0
    max_message_chars: int = 4096
    download_attachments: bool = True
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True
    require_signature: bool = True

    def __post_init__(self) -> None:
        if not self.phone_number_id.strip():
            raise ValueError("SORA_WHATSAPP_PHONE_NUMBER_ID must not be empty")
        if not self.access_token.strip():
            raise ValueError("SORA_WHATSAPP_ACCESS_TOKEN must not be empty")
        if not self.verify_token.strip():
            raise ValueError("SORA_WHATSAPP_VERIFY_TOKEN must not be empty")
        if self.require_signature and not self.app_secret.strip():
            raise ValueError(
                "SORA_WHATSAPP_APP_SECRET is required to verify webhook "
                "signatures; set it or explicitly allow unverified requests"
            )
        if not self.api_base.strip():
            raise ValueError("whatsapp api_base must not be empty")
        if not self.webhook_path.startswith("/"):
            raise ValueError("whatsapp webhook_path must start with /")
        if self.webhook_port < 0:
            raise ValueError("whatsapp webhook_port must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("whatsapp request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("whatsapp max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("whatsapp max_attachment_bytes must be > 0")


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    phone_number_id = env_str("SORA_WHATSAPP_PHONE_NUMBER_ID")
    access_token = env_str("SORA_WHATSAPP_ACCESS_TOKEN") or str(
        secrets.get("SORA_WHATSAPP_ACCESS_TOKEN", "")
    ).strip()
    verify_token = env_str("SORA_WHATSAPP_VERIFY_TOKEN") or str(
        secrets.get("SORA_WHATSAPP_VERIFY_TOKEN", "")
    ).strip()
    app_secret = env_str("SORA_WHATSAPP_APP_SECRET") or str(
        secrets.get("SORA_WHATSAPP_APP_SECRET", "")
    ).strip()
    if not phone_number_id:
        raise ValueError(
            "SORA_WHATSAPP_PHONE_NUMBER_ID is not set; copy it from the Meta "
            "app's WhatsApp > API Setup page"
        )
    if not access_token:
        raise ValueError(
            "SORA_WHATSAPP_ACCESS_TOKEN is not set; create a permanent token "
            "for the WhatsApp Business account"
        )
    if not verify_token:
        raise ValueError(
            "SORA_WHATSAPP_VERIFY_TOKEN is not set; choose a value and enter "
            "the same value in the Meta webhook configuration"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_WHATSAPP_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    allow_unverified = env_bool("SORA_WHATSAPP_ALLOW_UNVERIFIED", False)
    return WhatsAppConfig(
        phone_number_id=phone_number_id,
        access_token=access_token,
        verify_token=verify_token,
        app_secret=app_secret,
        allow_from=allow_from,
        api_base=env_str(
            "SORA_WHATSAPP_API_BASE",
            str(options.get("api_base") or "https://graph.facebook.com/v21.0"),
        ),
        webhook_host=env_str(
            "SORA_WHATSAPP_WEBHOOK_HOST",
            str(options.get("webhook_host") or "127.0.0.1"),
        ),
        webhook_port=env_int(
            "SORA_WHATSAPP_WEBHOOK_PORT",
            int(options.get("webhook_port") or 8080),
        ),
        webhook_path=env_str(
            "SORA_WHATSAPP_WEBHOOK_PATH",
            str(options.get("webhook_path") or "/whatsapp"),
        ),
        request_timeout=env_float("SORA_WHATSAPP_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_WHATSAPP_MAX_CHARS",
            int(options.get("max_message_chars") or 4096),
        ),
        download_attachments=env_bool("SORA_WHATSAPP_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_WHATSAPP_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
        require_signature=not allow_unverified,
    )
