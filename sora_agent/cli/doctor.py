"""Read-only health check for the saved settings and runtime paths."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys

from ..delegation import DelegationStore
from ..documents import default_registry
from ..model_catalog import format_tokens, provider_info
from ..runtime_setup import (
    approval_mode,
    build_memory,
    build_registry,
    feature_enabled,
    files_enabled,
    provider_kind,
    provider_label,
    quality_enabled,
    resolve_web_search,
    state_root_path,
    streaming_enabled,
)
from ..settings import Settings, config_path, secrets_path
from ..toolpack.config import ToolPackConfig
from . import ui


def run_doctor(settings: Settings | None, secrets: dict[str, str]) -> int:
    ui.banner("Sora", "☁️", _version(), "Health check")
    if settings is None:
        ui.warning(f"No configuration found at {config_path()}.")
        ui.info("Run `sora setup` to create one.")
        return 1

    kind = provider_kind(settings)
    info = provider_info(kind)
    env_key = os.getenv(info.key_env, "").strip() if info and info.key_env else ""
    saved_key = secrets.get(info.key_env, "").strip() if info and info.key_env else ""
    token_env = os.getenv("SORA_TELEGRAM_TOKEN", "").strip()
    token_saved = secrets.get("SORA_TELEGRAM_TOKEN", "").strip()

    ui.kv_table(
        "Configuration",
        [
            ("Path", str(config_path())),
            ("Mode", settings.setup.mode),
            ("Completed", settings.setup.completed_at or "unknown"),
            ("Secrets", str(secrets_path())),
        ],
    )
    ui.kv_table(
        "Model",
        [
            ("Provider", provider_label(settings)),
            ("Model", settings.provider.model),
            ("Temperature", f"{settings.provider.temperature:g}"),
            (
                "Context window",
                format_tokens(settings.provider.context_window),
            ),
            ("Base URL", settings.provider.base_url or "(provider default)"),
            (
                "API key",
                "(not required)"
                if kind in {"mock", "demo"}
                else (
                    "environment" if env_key else ("saved" if saved_key else "MISSING")
                ),
            ),
        ],
    )
    channel_kind = str(settings.channel.kind or "telegram").strip().lower()
    channel_names = {
        "desktop": "SoraDesk desktop app",
        "telegram": "Telegram",
        "discord": "Discord",
        "slack": "Slack",
        "whatsapp": "WhatsApp Cloud API",
        "teams": "Microsoft Teams",
        "signal": "Signal",
        "googlechat": "Google Chat",
        "matrix": "Matrix",
        "feishu": "Feishu / Lark",
        "imessage": "iMessage",
    }
    channel_keys = {
        "discord": ("SORA_DISCORD_TOKEN",),
        "slack": ("SORA_SLACK_BOT_TOKEN", "SORA_SLACK_APP_TOKEN"),
        "whatsapp": (
            "SORA_WHATSAPP_PHONE_NUMBER_ID",
            "SORA_WHATSAPP_ACCESS_TOKEN",
            "SORA_WHATSAPP_VERIFY_TOKEN",
            "SORA_WHATSAPP_APP_SECRET",
        ),
        "teams": ("SORA_TEAMS_APP_ID", "SORA_TEAMS_APP_PASSWORD"),
        "signal": ("SORA_SIGNAL_HTTP_URL", "SORA_SIGNAL_ACCOUNT"),
        "googlechat": (
            "SORA_GOOGLE_CHAT_WEBHOOK_URL",
            "SORA_GOOGLE_CHAT_AUDIENCE",
        ),
        "matrix": ("SORA_MATRIX_HOMESERVER", "SORA_MATRIX_ACCESS_TOKEN"),
        "feishu": (
            "SORA_FEISHU_APP_ID",
            "SORA_FEISHU_APP_SECRET",
            "SORA_FEISHU_VERIFICATION_TOKEN",
        ),
        "imessage": (
            "SORA_IMESSAGE_SERVER_URL",
            "SORA_IMESSAGE_PASSWORD",
            "SORA_IMESSAGE_WEBHOOK_SECRET",
        ),
    }
    channel_rows = [
        ("Kind", channel_names.get(channel_kind, channel_kind)),
        (
            "Allowed users",
            ", ".join(settings.channel.allow_from) or "none (denies all)",
        ),
    ]
    if channel_kind == "desktop":
        channel_rows.append(("Bridge", "ws://127.0.0.1:18791"))
    elif channel_kind == "telegram":
        channel_rows.extend(
            [
                (
                    "Telegram token",
                    "environment"
                    if token_env
                    else ("saved" if token_saved else "MISSING"),
                ),
                ("Parse mode", settings.channel.parse_mode or "plain text"),
            ]
        )
    else:
        missing = [
            key
            for key in channel_keys.get(channel_kind, ())
            if not (os.getenv(key, "").strip() or secrets.get(key, "").strip())
        ]
        channel_rows.append(
            ("Credentials", "MISSING: " + ", ".join(missing) if missing else "configured")
        )
    ui.kv_table("Channel", channel_rows)
    memory = settings.features.memory
    memory_status = _memory_status(settings, secrets)
    ui.kv_table(
        "Features",
        [
            ("Memory", memory_status),
            ("Files", "On" if files_enabled(settings) else "Off"),
            (
                "Core tools",
                "On" if feature_enabled(settings, "tools", "SORA_TOOLPACK") else "Off",
            ),
            (
                "Vision",
                settings.features.files.vision_model or "auto / OCR fallback",
            ),
            (
                "CortexDB sidecar",
                ("configured" if memory.enabled and memory.sidecar else "off"),
            ),
            ("Identity", "On" if settings.features.identity else "Off"),
            (
                "Session state",
                (
                    f"On ({state_root_path(settings)})"
                    if feature_enabled(settings, "state", "SORA_STATE")
                    else "Off"
                ),
            ),
            ("Clock", "On" if settings.features.clock else "Off"),
            ("Calendar", "On" if settings.features.calendar else "Off"),
            (
                "Sessions",
                "On" if settings.features.sessions else "In-memory only",
            ),
            ("Streaming", "On" if streaming_enabled(settings) else "Off"),
            ("Approval", approval_mode(settings)),
            ("Quality", "On" if quality_enabled(settings) else "Off"),
        ],
    )
    _tools_table(settings, secrets)
    _files_table()
    _delegation_table()
    _warnings(settings, secrets, memory_status)
    return 0


def _tools_table(settings: Settings, secrets: dict[str, str]) -> None:
    pack = ToolPackConfig.from_env()
    web = resolve_web_search(settings, secrets)
    provider = web.resolved_provider()
    if provider == "tavily":
        key_state = "saved" if web.tavily_api_key else "MISSING"
        search_label = f"Tavily ({key_state} key)"
    elif provider == "brave":
        key_state = "saved" if web.brave_api_key else "MISSING"
        search_label = f"Brave ({key_state} key)"
    elif provider == "custom":
        search_label = f"Custom ({web.url or 'no URL'})"
    else:
        search_label = "DuckDuckGo (no key)" if provider == "duckduckgo" else "Off"

    roots = ", ".join(str(item) for item in pack.allowed_roots())
    try:
        registry = build_registry(settings, secrets=secrets)
        names = sorted(tool.name for tool in registry.all())
        tools_state = f"{len(names)} registered"
    except Exception as exc:  # noqa: BLE001 - doctor must not crash
        names = []
        tools_state = f"error: {exc}"

    packages = [
        package
        for package in ("requests", "numpy")
        if importlib.util.find_spec(package) is not None
    ]

    ui.kv_table(
        "Tools",
        [
            ("Core pack", "On" if settings.features.tools else "Off"),
            ("Workspace", str(pack.root)),
            ("Allowed roots", roots),
            ("Confined", "Yes" if pack.restrict_to_root else "No"),
            ("Web search", search_label),
            ("Python", sys.version.split()[0]),
            ("Code extras", ", ".join(packages) or "stdlib only"),
            ("Registered", tools_state),
            (
                "File tools",
                ", ".join(
                    name
                    for name in ("read_file", "write_file", "patch", "search_files")
                    if name in names
                )
                or "none",
            ),
        ],
    )


def _files_table() -> None:
    converters = ", ".join(default_registry().capabilities())
    pdf = (
        "available (pypdf)"
        if importlib.util.find_spec("pypdf") is not None
        else "text layer needs the files extra (pypdf)"
    )
    ocr = (
        "available (pytesseract + Pillow)"
        if importlib.util.find_spec("pytesseract") is not None
        and importlib.util.find_spec("PIL") is not None
        else "not installed (vision model may be used for images)"
    )
    ui.kv_table(
        "File understanding",
        [
            ("Converters", converters),
            ("PDF text", pdf),
            ("Image OCR", ocr),
            ("Scanned PDFs", "text layer only; OCR/raster fallback not wired"),
            ("Legacy Office", ".doc/.xls/.ppt are not converted"),
        ],
    )


def _delegation_table() -> None:
    store = DelegationStore()
    agents = store.list(include_paused=True)
    active = sum(1 for agent in agents if agent.status == "active")
    ui.kv_table(
        "Specialists",
        [
            ("Store", str(store.root)),
            ("Agents", f"{active} active / {len(agents)} total"),
        ],
    )


def _memory_status(settings: Settings, secrets: dict[str, str]) -> str:
    env_memory = os.getenv("SORA_MEMORY", "").strip().lower()
    if env_memory and env_memory not in {"1", "true", "yes", "on"}:
        return "Off (SORA_MEMORY=0)"
    try:
        memory = build_memory(
            settings, None, secrets, auto_dream=False
        )
    except Exception as exc:  # noqa: BLE001 - doctor must still render
        return f"Unavailable ({type(exc).__name__})"
    if memory is None:
        return "Off"
    try:
        status = memory.availability(probe=True)
    except Exception:  # noqa: BLE001 - a probe failure is the result
        return "Unavailable (probe failed)"
    finally:
        with contextlib.suppress(Exception):
            memory.close()
    mode = str(status.get("mode", ""))
    if mode == "daily_only":
        return "Daily-only (long-term off)"
    if mode == "daily_only_fallback":
        return "Daily-only fallback (long-term unavailable)"
    if mode == "long_term" and status.get("embedder") is False:
        return "Long-term (lexical; sidecar has no embedder)"
    return "Long-term (available)" if status.get("longterm_available") else "Unavailable"


def _warnings(
    settings: Settings,
    secrets: dict[str, str],
    memory_status: str = "",
) -> None:
    problems: list[str] = []
    kind = provider_kind(settings)
    if kind not in {"mock", "demo"}:
        info = provider_info(kind)
        if (
            info is not None
            and not os.getenv(info.key_env, "").strip()
            and not secrets.get(info.key_env, "").strip()
        ):
            problems.append(f"{info.key_env} is not configured.")
    if (
        not os.getenv("SORA_TELEGRAM_TOKEN", "").strip()
        and not secrets.get("SORA_TELEGRAM_TOKEN", "").strip()
    ):
        problems.append("Telegram bot token is not configured.")
    if not settings.channel.allow_from:
        problems.append(
            "Telegram allow list is empty; the gateway will ignore everyone."
        )
    web = resolve_web_search(settings, secrets)
    provider = web.resolved_provider()
    if feature_enabled(settings, "tools", "SORA_TOOLPACK"):
        if provider == "off":
            problems.append("Web search is off (web_search returns an error).")
        elif provider == "tavily" and not web.tavily_api_key:
            problems.append("Web search uses Tavily but TAVILY_API_KEY is missing.")
        elif provider == "brave" and not web.brave_api_key:
            problems.append("Web search uses Brave but BRAVE_SEARCH_API_KEY is missing.")
        elif provider == "custom" and not web.url:
            problems.append("Web search is set to custom but has no endpoint URL.")
    if settings.features.memory.enabled and "unavailable" in memory_status.lower():
        problems.append(
            "Long-term memory is enabled but unavailable; daily-only consolidation "
            "will run until the CortexDB sidecar is healthy."
        )
    if approval_mode(settings) == "off":
        problems.append("Approval is OFF: consequential tools execute without asking.")
    if not quality_enabled(settings):
        problems.append("Quality evaluation is OFF: /eval has no report.")
    if not problems:
        ui.success("No problems found.")
        return
    ui.blank()
    for problem in problems:
        ui.warning(problem)


def _version() -> str:
    from .. import __version__

    return __version__
