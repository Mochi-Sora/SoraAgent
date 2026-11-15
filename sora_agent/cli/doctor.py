"""Read-only health check for the saved settings and runtime paths."""

from __future__ import annotations

import importlib.util
import os
import sys

from ..model_catalog import format_tokens, provider_info
from ..runtime_setup import (
    approval_mode,
    build_registry,
    feature_enabled,
    files_enabled,
    memory_enabled,
    provider_kind,
    provider_label,
    quality_enabled,
    resolve_web_search,
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
    ui.kv_table(
        "Channel",
        [
            (
                "Telegram token",
                "environment" if token_env else ("saved" if token_saved else "MISSING"),
            ),
            (
                "Allowed users",
                ", ".join(settings.channel.allow_from) or "none (denies all)",
            ),
            ("Parse mode", settings.channel.parse_mode or "plain text"),
        ],
    )
    memory = settings.features.memory
    ui.kv_table(
        "Features",
        [
            ("Memory", "On" if memory_enabled(settings) else "Off"),
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
    _warnings(settings, secrets)
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


def _warnings(settings: Settings, secrets: dict[str, str]) -> None:
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
