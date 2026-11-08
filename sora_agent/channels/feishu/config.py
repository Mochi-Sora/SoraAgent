"""Feishu / Lark channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class FeishuConfig:
    app_id: str
    app_secret: str
    verification_token: str
    allow_from: tuple[str, ...] = ()
    domain: str = "feishu"
    api_base: str = ""
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8080
    webhook_path: str = "/feishu"
    request_timeout: float = 30.0
    max_message_chars: int = 4000
    download_attachments: bool = True
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True
    reaction_emoji: str = "OnIt"

    def __post_init__(self) -> None:
        if not self.app_id.strip():
            raise ValueError("SORA_FEISHU_APP_ID must not be empty")
        if not self.app_secret.strip():
            raise ValueError("SORA_FEISHU_APP_SECRET must not be empty")
        if not self.verification_token.strip():
            raise ValueError(
                "SORA_FEISHU_VERIFICATION_TOKEN must not be empty; copy it "
                "from the app's Event Subscriptions page"
            )
        if self.domain not in {"feishu", "lark"}:
            raise ValueError("feishu domain must be feishu or lark")
        if not self.webhook_path.startswith("/"):
            raise ValueError("feishu webhook_path must start with /")
        if self.webhook_port < 0:
            raise ValueError("feishu webhook_port must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("feishu request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("feishu max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("feishu max_attachment_bytes must be > 0")

    @property
    def resolved_api_base(self) -> str:
        if self.api_base.strip():
            return self.api_base.strip().rstrip("/")
        if self.domain == "lark":
            return "https://open.larksuite.com"
        return "https://open.feishu.cn"


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    app_id = env_str("SORA_FEISHU_APP_ID")
    app_secret = env_str("SORA_FEISHU_APP_SECRET") or str(
        secrets.get("SORA_FEISHU_APP_SECRET", "")
    ).strip()
    verification_token = env_str("SORA_FEISHU_VERIFICATION_TOKEN") or str(
        secrets.get("SORA_FEISHU_VERIFICATION_TOKEN", "")
    ).strip()
    if not app_id:
        raise ValueError(
            "SORA_FEISHU_APP_ID is not set; create a custom app in the "
            "Feishu/Lark developer console"
        )
    if not app_secret:
        raise ValueError("SORA_FEISHU_APP_SECRET is not set")
    if not verification_token:
        raise ValueError(
            "SORA_FEISHU_VERIFICATION_TOKEN is not set; copy it from the "
            "app's Event Subscriptions page"
        )
    if env_str("SORA_FEISHU_ENCRYPT_KEY"):
        raise ValueError(
            "Feishu encrypt_key is not supported yet; disable payload "
            "encryption in the app's Event Subscriptions settings"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_FEISHU_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return FeishuConfig(
        app_id=app_id,
        app_secret=app_secret,
        verification_token=verification_token,
        allow_from=allow_from,
        domain=env_str(
            "SORA_FEISHU_DOMAIN", str(options.get("domain") or "feishu")
        ).lower(),
        api_base=env_str(
            "SORA_FEISHU_API_BASE", str(options.get("api_base") or "")
        ),
        webhook_host=env_str(
            "SORA_FEISHU_WEBHOOK_HOST",
            str(options.get("webhook_host") or "127.0.0.1"),
        ),
        webhook_port=env_int(
            "SORA_FEISHU_WEBHOOK_PORT",
            int(options.get("webhook_port") or 8080),
        ),
        webhook_path=env_str(
            "SORA_FEISHU_WEBHOOK_PATH",
            str(options.get("webhook_path") or "/feishu"),
        ),
        request_timeout=env_float("SORA_FEISHU_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_FEISHU_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        download_attachments=env_bool("SORA_FEISHU_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_FEISHU_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
        reaction_emoji=env_str(
            "SORA_FEISHU_REACTION_EMOJI",
            str(options.get("reaction_emoji") or "OnIt"),
        ),
    )
