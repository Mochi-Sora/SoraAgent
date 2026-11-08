"""Microsoft Teams channel configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._util import env_bool, env_float, env_int, env_str, split_ids


@dataclass(frozen=True)
class TeamsConfig:
    app_id: str
    app_password: str
    tenant_id: str = "common"
    allow_from: tuple[str, ...] = ()
    token_url: str = ""
    openid_config_url: str = (
        "https://login.botframework.com/v1/.well-known/openidconfiguration"
    )
    scope: str = "https://api.botframework.com/.default"
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8080
    webhook_path: str = "/teams"
    request_timeout: float = 30.0
    max_message_chars: int = 4000
    download_attachments: bool = True
    max_attachment_bytes: int = 20 * 1024 * 1024
    typing: bool = True
    verify_jwt: bool = True

    def __post_init__(self) -> None:
        if not self.app_id.strip():
            raise ValueError("SORA_TEAMS_APP_ID must not be empty")
        if not self.app_password.strip():
            raise ValueError("SORA_TEAMS_APP_PASSWORD must not be empty")
        if not self.tenant_id.strip():
            raise ValueError("SORA_TEAMS_TENANT_ID must not be empty")
        if not self.webhook_path.startswith("/"):
            raise ValueError("teams webhook_path must start with /")
        if self.webhook_port < 0:
            raise ValueError("teams webhook_port must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("teams request_timeout must be > 0")
        if self.max_message_chars <= 0:
            raise ValueError("teams max_message_chars must be > 0")
        if self.max_attachment_bytes <= 0:
            raise ValueError("teams max_attachment_bytes must be > 0")

    @property
    def resolved_token_url(self) -> str:
        if self.token_url.strip():
            return self.token_url.strip()
        return (
            f"https://login.microsoftonline.com/{self.tenant_id}"
            "/oauth2/v2.0/token"
        )


def from_settings(settings, secrets, args):
    options: dict[str, Any] = dict(getattr(settings.channel, "options", {}) or {})
    app_id = env_str("SORA_TEAMS_APP_ID")
    app_password = env_str("SORA_TEAMS_APP_PASSWORD") or str(
        secrets.get("SORA_TEAMS_APP_PASSWORD", "")
    ).strip()
    if not app_id:
        raise ValueError(
            "SORA_TEAMS_APP_ID is not set; register a bot in Azure Bot Service"
        )
    if not app_password:
        raise ValueError(
            "SORA_TEAMS_APP_PASSWORD is not set; copy the client secret from "
            "the bot's app registration"
        )
    allow_flag = getattr(args, "allow", None)
    if allow_flag is not None:
        allow_from = split_ids(allow_flag)
    else:
        allow_env = env_str("SORA_TEAMS_ALLOW_FROM")
        allow_from = (
            split_ids(allow_env)
            if allow_env
            else tuple(settings.channel.allow_from)
        )
    return TeamsConfig(
        app_id=app_id,
        app_password=app_password,
        tenant_id=env_str(
            "SORA_TEAMS_TENANT_ID", str(options.get("tenant_id") or "common")
        ),
        allow_from=allow_from,
        token_url=env_str("SORA_TEAMS_TOKEN_URL", ""),
        openid_config_url=env_str(
            "SORA_TEAMS_OPENID_CONFIG_URL",
            str(
                options.get("openid_config_url")
                or "https://login.botframework.com/v1/.well-known/openidconfiguration"
            ),
        ),
        scope=env_str(
            "SORA_TEAMS_SCOPE",
            str(options.get("scope") or "https://api.botframework.com/.default"),
        ),
        webhook_host=env_str(
            "SORA_TEAMS_WEBHOOK_HOST",
            str(options.get("webhook_host") or "127.0.0.1"),
        ),
        webhook_port=env_int(
            "SORA_TEAMS_WEBHOOK_PORT",
            int(options.get("webhook_port") or 8080),
        ),
        webhook_path=env_str(
            "SORA_TEAMS_WEBHOOK_PATH",
            str(options.get("webhook_path") or "/teams"),
        ),
        request_timeout=env_float("SORA_TEAMS_REQUEST_TIMEOUT", 30.0),
        max_message_chars=env_int(
            "SORA_TEAMS_MAX_CHARS",
            int(options.get("max_message_chars") or 4000),
        ),
        download_attachments=env_bool("SORA_TEAMS_DOWNLOAD_FILES", True),
        max_attachment_bytes=env_int(
            "SORA_TEAMS_MAX_ATTACHMENT_BYTES", 20 * 1024 * 1024
        ),
        typing=bool(getattr(settings.channel, "typing", True)),
        verify_jwt=not env_bool("SORA_TEAMS_ALLOW_UNVERIFIED", False),
    )
