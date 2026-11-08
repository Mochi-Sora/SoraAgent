"""The two-path setup wizard.

Recommended Setup is deliberately short: channel, main model, and a few agent
settings, then a compact review.  Advanced Setup walks through every subsystem,
starts with all optional features off, and requires explicit acknowledgement of
the safety implications.
"""

from __future__ import annotations

import copy
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Callable

from .. import __version__
from ..model_catalog import (
    FALLBACK_CONTEXT_WINDOW,
    format_tokens,
    provider_info,
    providers,
)
from ..settings import (
    MODES,
    MemorySettings,
    Settings,
    SettingsError,
    advanced_settings,
    config_path,
    load_secrets,
    load_settings,
    recommended_settings,
    save_secrets,
    save_settings,
    secrets_path,
)
from . import prompts, ui
from .prompts import WizardCancelled

_PERSONALITY_LABELS = {
    "professional": "Professional",
    "friendly": "Friendly",
    "candid": "Candid",
    "efficient": "Efficient",
    "neutral": "Neutral / off",
}
_WARMTH_LABELS = {
    "very_cold": "Very cold",
    "cold": "Cold",
    "neutral": "Neutral",
    "warm": "Warm",
    "very_warm": "Very warm",
}
_APPROVAL_LABELS = {
    "off": "Off — consequential tools execute without asking",
    "ask": "Ask — prompt before consequential tools",
}


def run_wizard(mode: str | None = None) -> bool:
    """Run setup and return whether a configuration was saved."""
    if mode is not None:
        mode = mode.lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
    try:
        ui.clear()
        existing = load_settings()
        secrets = load_secrets()
        _main_banner()
        if mode is None:
            mode = _choose_mode(existing)
        if mode == "exit" or mode is None:
            ui.info("Nothing changed.")
            return False
        if mode == "edit" and existing is not None:
            _edit_existing(existing, secrets)
            return True
        if mode == "recommended":
            settings = _recommended_start(existing)
            _recommended_flow(settings, secrets)
        else:
            settings = _advanced_start(existing)
            _advanced_flow(settings, secrets)
        return True
    except WizardCancelled:
        ui.blank()
        ui.warning("Setup cancelled. Nothing was saved.")
        return False
    except SettingsError as exc:
        ui.error(str(exc))
        return False


# ---------------------------------------------------------------------------
# Entry screens
# ---------------------------------------------------------------------------


def _main_banner() -> None:
    ui.banner(
        "Sora",
        "☁️",
        __version__,
        "A model-agnostic agent with real approval and quality gates.",
    )


def _choose_mode(existing: Settings | None) -> str:
    choices: list[tuple[str, str]] = [
        (
            "[R] Recommended Setup — everything on, tuned for daily use",
            "recommended",
        ),
        (
            "[A] Advanced Setup (tinker with caution) — everything off",
            "advanced",
        ),
    ]
    if existing is not None:
        choices.append(("[E] Edit current configuration", "edit"))
    choices.append(("[X] Exit", "exit"))
    return prompts.select(
        "How would you like to configure Sora?",
        choices,
        default="recommended",
    )


def _recommended_start(existing: Settings | None) -> Settings:
    settings = recommended_settings()
    if existing is not None and existing.setup.mode == "recommended":
        # Keep the personal choices; restore the recommended feature set.
        settings.agent = copy.deepcopy(existing.agent)
        settings.provider = copy.deepcopy(existing.provider)
        settings.channel = copy.deepcopy(existing.channel)
        settings.paths = copy.deepcopy(existing.paths)
        if existing.setup.completed_at:
            settings.setup.completed_at = existing.setup.completed_at
    settings.setup.mode = "recommended"
    return settings


def _advanced_start(existing: Settings | None) -> Settings:
    settings = advanced_settings()
    if existing is not None and existing.setup.mode == "advanced":
        settings.agent = copy.deepcopy(existing.agent)
        settings.provider = copy.deepcopy(existing.provider)
        settings.channel = copy.deepcopy(existing.channel)
        settings.paths = copy.deepcopy(existing.paths)
    settings.setup.mode = "advanced"
    return settings


# ---------------------------------------------------------------------------
# Recommended flow
# ---------------------------------------------------------------------------


def _recommended_flow(settings: Settings, secrets: dict[str, str]) -> None:
    ui.section(
        "Recommended Setup",
        "Everything is enabled and tuned for daily use. Only the channel, "
        "main model, and a few agent details are asked here.",
    )
    _configure_telegram(settings, secrets, advanced=False)
    _configure_provider(settings, secrets, advanced=False)
    _configure_agent(settings, minimal=True)
    _recommended_review(settings, secrets)
    _finish(settings, secrets)


def _recommended_review(
    settings: Settings, secrets: dict[str, str]
) -> None:
    while True:
        ui.clear()
        _main_banner()
        ui.section("Review", "These are the settings that will be saved.")
        _show_recommended_settings(settings, secrets)
        choice = prompts.select(
            "What would you like to do?",
            [
                ("[C] Save and finish", "save"),
                ("[N] Edit agent name, icon, temperature, or personality", "agent"),
                ("[T] Edit Telegram channel", "channel"),
                ("[M] Edit main model", "model"),
                ("[X] Cancel", "cancel"),
            ],
            default="save",
        )
        if choice == "save":
            return
        if choice == "cancel":
            raise WizardCancelled()
        if choice == "agent":
            _configure_agent(settings, minimal=True)
        elif choice == "channel":
            _configure_telegram(settings, secrets, advanced=False)
        elif choice == "model":
            _configure_provider(settings, secrets, advanced=False)


def _show_recommended_settings(
    settings: Settings, secrets: dict[str, str]
) -> None:
    agent = settings.agent
    ui.kv_table(
        "Agent",
        [
            ("Name", agent.name),
            ("Icon", agent.icon or "(none)"),
            ("Personality", _personality_label(agent.personality.base)),
            ("Warmth", _WARMTH_LABELS.get(agent.personality.warmth, agent.personality.warmth)),
            ("Temperature", f"{settings.provider.temperature:g}"),
        ],
    )
    ui.kv_table(
        "Main model",
        [
            ("Provider", _provider_label(settings.provider.kind)),
            ("Model", settings.provider.model),
            (
                "Context window",
                format_tokens(settings.provider.context_window),
            ),
            ("Base URL", settings.provider.base_url or "(provider default)"),
            (
                "API key",
                _key_status(settings.provider.kind, secrets),
            ),
        ],
    )
    ui.kv_table(
        "Channel",
        [
            (
                "Telegram token",
                _token_status(secrets),
            ),
            (
                "Allowed users",
                ", ".join(settings.channel.allow_from) or "(none)",
            ),
        ],
    )
    ui.kv_table(
        "Applied automatically",
        [
            ("Memory", "Full architecture, including CortexDB sidecar"),
            ("Files", "On — attachments become markdown (20 MB max)"),
            ("Identity", "On"),
            ("Clock", "On"),
            ("Calendar", "On"),
            ("Sessions", "On"),
            ("Tools", "Demo pack"),
            ("Approvals", "Ask before consequential tools"),
            ("Quality gate", "On"),
            ("Limits", "12 steps / 80 turns / 60 s / 3 retries"),
        ],
        footer="Use Advanced Setup when you want to change these individually.",
    )


# ---------------------------------------------------------------------------
# Advanced flow
# ---------------------------------------------------------------------------


def _advanced_flow(settings: Settings, secrets: dict[str, str]) -> None:
    _advanced_warning()
    ui.section(
        "Advanced Setup",
        "Every optional subsystem starts off. Each section explains what it "
        "changes before you decide.",
    )
    _configure_telegram(settings, secrets, advanced=True)
    _configure_provider(settings, secrets, advanced=True)
    _configure_agent(settings, minimal=False)
    _configure_memory(settings)
    _configure_files(settings)
    _configure_identity(settings)
    _configure_clock_calendar(settings)
    _configure_sessions(settings)
    _configure_tools_and_limits(settings)
    _configure_approval(settings)
    _configure_quality(settings)
    _advanced_review(settings, secrets)
    _advanced_final_confirmation(settings)
    _finish(settings, secrets)


def _advanced_warning() -> None:
    ui.clear()
    _main_banner()
    ui.warning_panel(
        "Advanced Setup (tinker with caution)",
        [
            "This path turns off every optional subsystem by default:",
            "  • memory (including the CortexDB sidecar)",
            "  • identity (USER.md / SOUL.md)",
            "  • file understanding, clock, calendar, and session persistence",
            "  • tools, approval prompts, and quality evaluation",
            "",
            "Consequential tools can execute without confirmation.",
            "Runs will produce no quality report.",
            "",
            "Every setting can be enabled individually later, but you are",
            "responsible for understanding the consequences.",
        ],
    )
    typed = prompts.text(
        "Type `I understand` to continue",
        "",
        validate=lambda value: (
            True
            if value.strip().lower() == "i understand"
            else "type `I understand` exactly"
        ),
    )
    if typed.strip().lower() != "i understand":
        raise WizardCancelled()


def _advanced_review(settings: Settings, secrets: dict[str, str]) -> None:
    while True:
        ui.clear()
        _main_banner()
        ui.section("Full review", "Every section is editable before saving.")
        _show_advanced_settings(settings, secrets)
        choice = prompts.select(
            "What would you like to do?",
            [
                ("[C] Save and finish", "save"),
                ("[E] Edit a section", "edit"),
                ("[R] Reset optional features to Advanced defaults", "reset"),
                ("[X] Cancel", "cancel"),
            ],
            default="save",
        )
        if choice == "save":
            return
        if choice == "cancel":
            raise WizardCancelled()
        if choice == "reset":
            _reset_advanced_features(settings)
            continue
        section = prompts.select(
            "Which section?",
            [
                ("Channel", "channel"),
                ("Provider and credentials", "provider"),
                ("Agent name, icon, personality, temperature", "agent"),
                ("Memory", "memory"),
                ("Files and images", "files"),
                ("Identity", "identity"),
                ("Clock and calendar", "clock"),
                ("Sessions", "sessions"),
                ("Tools and limits", "tools"),
                ("Approval", "approval"),
                ("Quality evaluation", "quality"),
                ("<- Back", "back"),
            ],
            default="back",
        )
        if section == "back":
            continue
        dispatch: dict[str, Callable[[], None]] = {
            "channel": lambda: _configure_telegram(
                settings, secrets, advanced=True
            ),
            "provider": lambda: _configure_provider(
                settings, secrets, advanced=True
            ),
            "agent": lambda: _configure_agent(settings, minimal=False),
            "memory": lambda: _configure_memory(settings),
            "files": lambda: _configure_files(settings),
            "identity": lambda: _configure_identity(settings),
            "clock": lambda: _configure_clock_calendar(settings),
            "sessions": lambda: _configure_sessions(settings),
            "tools": lambda: _configure_tools_and_limits(settings),
            "approval": lambda: _configure_approval(settings),
            "quality": lambda: _configure_quality(settings),
        }
        dispatch[section]()


def _advanced_final_confirmation(settings: Settings) -> None:
    ui.blank()
    ui.warning_panel(
        "Final safety status",
        [
            f"Approval: {'OFF' if settings.features.approval == 'off' else 'ASK'}",
            f"Quality:  {'OFF' if not settings.features.quality else 'ON'}",
            f"Memory:   {'ON' if settings.features.memory.enabled else 'OFF'}",
            f"Files:    {'ON' if settings.features.files.enabled else 'OFF'}",
            f"Identity: {'ON' if settings.features.identity else 'OFF'}",
            "",
            "This is the configuration that will be written to disk.",
        ],
    )
    typed = prompts.text(
        "Type `save` to write this configuration (or press Ctrl-C to cancel)",
        default="save",
        validate=lambda value: (
            True if value.strip().lower() == "save" else "type `save`"
        ),
    )
    if typed.strip().lower() != "save":
        raise WizardCancelled()


def _reset_advanced_features(settings: Settings) -> None:
    preset = advanced_settings()
    settings.features = copy.deepcopy(preset.features)
    settings.limits = copy.deepcopy(preset.limits)
    settings.agent.personality = copy.deepcopy(preset.agent.personality)
    settings.provider.temperature = preset.provider.temperature
    ui.success(
        "Optional features, limits, personality, and temperature reset to the "
        "Advanced defaults. Channel and model were kept."
    )
    prompts.pause()


# ---------------------------------------------------------------------------
# Shared configuration steps
# ---------------------------------------------------------------------------


def _configure_telegram(
    settings: Settings, secrets: dict[str, str], *, advanced: bool
) -> None:
    ui.section(
        "Channel · Telegram",
        "Create a bot with @BotFather, then paste its token here. The token is "
        "stored in secrets.env, never in config.json.",
    )
    env_token = os.getenv("SORA_TELEGRAM_TOKEN", "").strip()
    current = secrets.get("SORA_TELEGRAM_TOKEN", "").strip()
    token = current or env_token
    if env_token:
        ui.info(
            "SORA_TELEGRAM_TOKEN is set in the environment. Environment "
            "variables override the saved value at runtime."
        )
    if current and not env_token:
        ui.info(f"Current token: {_mask(current)}")
        if prompts.confirm("Replace the Telegram bot token?", default=False):
            typed = prompts.password(
                "Telegram bot token", allow_blank=True
            )
            if typed:
                token = typed
    elif not env_token:
        token = prompts.password("Telegram bot token", allow_blank=False)
    if token and token != env_token:
        secrets["SORA_TELEGRAM_TOKEN"] = token

    default_allow = ", ".join(settings.channel.allow_from)
    raw_allow = prompts.text(
        "Allowed numeric Telegram user IDs (comma separated, * for everyone)",
        default=default_allow,
        validate=_validate_allow,
        instruction="Use @userinfobot to find your numeric id.",
    )
    settings.channel.allow_from = tuple(
        part.strip() for part in raw_allow.split(",") if part.strip()
    )
    settings.channel.kind = "telegram"
    if not settings.channel.allow_from:
        ui.warning(
            "The allow list is empty. Telegram messages will be ignored until "
            "you add your numeric user id."
        )

    if advanced:
        parse_value = prompts.select(
            "Parse mode",
            [
                ("None (plain text)", None),
                ("HTML", "HTML"),
                ("Markdown", "Markdown"),
                ("MarkdownV2", "MarkdownV2"),
            ],
            default=settings.channel.parse_mode,
        )
        settings.channel.parse_mode = parse_value
        settings.channel.api_base = prompts.text(
            "Telegram API base URL", settings.channel.api_base
        )
        settings.channel.poll_timeout = _ask_int(
            "Long-poll timeout (seconds)", settings.channel.poll_timeout, 0
        )
        settings.channel.request_timeout = _ask_float(
            "HTTP request timeout (seconds)",
            settings.channel.request_timeout,
            1.0,
        )
        settings.channel.max_message_chars = _ask_int(
            "Maximum characters per message",
            settings.channel.max_message_chars,
            1,
        )

    if token and prompts.confirm(
        "Verify the token with Telegram getMe?", default=True
    ):
        try:
            username = _fetch_bot_username(
                token, settings.channel.api_base
            )
            ui.success(f"Telegram bot verified: @{username}")
            ui.info(
                "Telegram display name and photo are managed with @BotFather; "
                "the ☁️ icon is used by Sora's own terminal UI."
            )
        except Exception as exc:  # noqa: BLE001 - network is optional
            ui.warning(f"Could not verify the token now: {exc}")
            ui.info("You can continue; the gateway will report auth errors.")


def _configure_provider(
    settings: Settings, secrets: dict[str, str], *, advanced: bool
) -> None:
    ui.section(
        "Main model",
        "Choose the provider and model that will answer and call tools.",
    )
    choices = [
        (f"{provider.label}", provider.id) for provider in providers()
    ]
    choices.append(("Mock (offline demo)", "mock"))
    previous_kind = settings.provider.kind
    kind = prompts.select(
        "Provider", choices, default=previous_kind
    )
    settings.provider.kind = kind
    if kind != previous_kind:
        settings.provider.base_url = None

    if kind == "mock":
        settings.provider.model = "mock"
        settings.provider.base_url = None
        settings.provider.context_window = FALLBACK_CONTEXT_WINDOW
        # Keep the preset temperature: 0.7 Recommended, 0.1 Advanced.
        ui.info("Mock mode needs no API key and never calls the network.")
        return

    info = provider_info(kind)
    if info is None:
        raise SettingsError(f"unknown provider kind: {kind}")

    if kind == "custom":
        settings.provider.base_url = prompts.text(
            "OpenAI-compatible base URL (for example https://host/v1)",
            settings.provider.base_url or "",
            validate=lambda value: (
                True
                if value.strip().startswith(("http://", "https://"))
                else "enter an http:// or https:// URL"
            ),
        )
    elif advanced:
        default_base = (
            settings.provider.base_url
            or os.getenv(info.base_env, "").strip()
            or info.default_base_url
            or ""
        )
        base = prompts.text(
            "Base URL (blank = provider default)",
            default_base,
        )
        settings.provider.base_url = base.strip() or None

    model_choices = [(item.display, item.id) for item in info.models]
    known_models = {value for _, value in model_choices}
    if (
        settings.provider.model
        and settings.provider.model not in known_models
        and settings.provider.model != "mock"
    ):
        model_choices.insert(
            0,
            (
                f"{settings.provider.model} (current, not in catalog)",
                settings.provider.model,
            ),
        )
        known_models.add(settings.provider.model)
    if model_choices and not advanced:
        model_choices.append(("Other (type a model id)", "__other__"))
        selected = prompts.select(
            "Model",
            model_choices,
            default=settings.provider.model,
        )
        if selected == "__other__":
            settings.provider.model = prompts.text(
                "Model ID",
                settings.provider.model or info.default_model,
                validate=_validate_nonempty,
            )
        else:
            settings.provider.model = selected
    else:
        settings.provider.model = prompts.text(
            "Model ID",
            default=(
                settings.provider.model
                if settings.provider.model not in {"", "mock"}
                else info.default_model
            ),
            validate=_validate_nonempty,
        )

    detected = None
    if info.models:
        from ..model_catalog import context_window_for

        detected = context_window_for(kind, settings.provider.model)
    if detected is not None and not advanced:
        settings.provider.context_window = detected
    else:
        default_window = (
            detected
            or (
                settings.provider.context_window
                if settings.provider.model in {
                    item.id for item in info.models
                }
                else None
            )
            or FALLBACK_CONTEXT_WINDOW
        )
        settings.provider.context_window = _ask_int(
            "Context window in tokens",
            default_window,
            1_000,
        )

    _configure_key(settings, secrets, kind, advanced=advanced)

    if advanced:
        settings.limits.provider_timeout = _ask_float(
            "Provider request timeout (seconds)",
            settings.limits.provider_timeout,
            1.0,
        )
        settings.limits.provider_retries = _ask_int(
            "Provider retries for transient failures",
            settings.limits.provider_retries,
            0,
        )
        if prompts.confirm(
            "Send one tiny test request to verify the connection?",
            default=False,
        ):
            _test_provider(settings, secrets)


def _configure_key(
    settings: Settings,
    secrets: dict[str, str],
    kind: str,
    *,
    advanced: bool,
) -> None:
    info = provider_info(kind)
    if info is None:
        return
    env_key = os.getenv(info.key_env, "").strip()
    stored = secrets.get(info.key_env, "").strip()
    if env_key:
        ui.info(
            f"{info.key_env} is set in the environment; it will override any "
            "saved key at runtime."
        )
        if not advanced:
            return
        if not prompts.confirm(
            "Save a different key anyway (the environment still wins)?",
            default=False,
        ):
            return
    elif stored:
        ui.info(f"Current {info.key_env}: {_mask(stored)}")
        if not prompts.confirm("Replace the API key?", default=False):
            return
    typed = prompts.password(
        f"{info.key_env} (leave blank to configure later)",
        allow_blank=True,
    )
    if typed:
        secrets[info.key_env] = typed
        ui.success(f"API key saved to {secrets_path()}.")
    else:
        ui.warning(
            "No API key provided. The provider will fail until you set "
            f"{info.key_env} or run `sora set-key`."
        )


def _test_provider(settings: Settings, secrets: dict[str, str]) -> None:
    try:
        from ..providers import Message
        from ..runtime_setup import build_provider

        provider = build_provider(settings, secrets)
        with ui.console.status("Testing provider connection..."):
            turn = provider.complete_turn(
                [Message("user", "Reply with the single word: ok")], []
            )
        text = (turn.text or "").strip()
        ui.success(f"Provider responded: {text[:60] or '(empty response)'}")
    except Exception as exc:  # noqa: BLE001 - a failed test is not fatal
        ui.warning(f"Connection test failed: {exc}")
        ui.info("You can still save and fix the credential later.")


def _configure_agent(settings: Settings, *, minimal: bool) -> None:
    ui.section(
        "Agent",
        "The name and icon are display-only. Personality is prompt guidance, "
        "not an automatic rewrite of SOUL.md.",
    )
    settings.agent.name = prompts.text(
        "Agent name", settings.agent.name
    )
    settings.agent.icon = prompts.text(
        "Display icon (emoji or short text, blank to hide)",
        settings.agent.icon,
    )
    if not minimal:
        enable = prompts.confirm(
            "Enable a personality prompt layer?",
            default=settings.agent.personality.enabled,
        )
        if not enable:
            settings.agent.personality.base = "neutral"
            settings.agent.personality.warmth = "neutral"
        else:
            _ask_personality(settings)
    else:
        _ask_personality(settings)

    settings.provider.temperature = _ask_float(
        "Temperature (0.0–2.0)",
        settings.provider.temperature,
        0.0,
        maximum=2.0,
    )


def _ask_personality(settings: Settings) -> None:
    base_choices = [
        ("Professional", "professional"),
        ("Friendly", "friendly"),
        ("Candid", "candid"),
        ("Efficient", "efficient"),
    ]
    settings.agent.personality.base = prompts.select(
        "Personality",
        base_choices,
        default=(
            settings.agent.personality.base
            if settings.agent.personality.base in dict(base_choices).values()
            else "friendly"
        ),
    )
    settings.agent.personality.warmth = prompts.select(
        "Warmth",
        [
            ("Very cold", "very_cold"),
            ("Cold", "cold"),
            ("Neutral", "neutral"),
            ("Warm", "warm"),
            ("Very warm", "very_warm"),
        ],
        default=settings.agent.personality.warmth,
    )


def _configure_memory(settings: Settings) -> None:
    memory = settings.features.memory
    ui.section(
        "Memory",
        "Long-term memory is entirely optional. When it is off, no daily "
        "transcript and no CortexDB database are written.",
    )
    enabled = prompts.confirm(
        "Enable the memory architecture?",
        default=memory.enabled,
    )
    if not enabled:
        settings.features.memory = MemorySettings(enabled=False)
        ui.info("Memory is off. The agent still has the current conversation.")
        return

    memory.enabled = True
    memory.sidecar = prompts.confirm(
        "Use the CortexDB sidecar for long-term memory?",
        default=memory.sidecar,
    )
    memory.auto_download = prompts.confirm(
        "Download the CortexDB sidecar automatically when it is missing?",
        default=memory.auto_download,
    )
    memory.dream = prompts.confirm(
        "Run nightly dream consolidation?",
        default=memory.dream,
    )
    memory.backup = prompts.confirm(
        "Keep backups before each dream run?",
        default=memory.backup,
    )
    memory.wake = prompts.confirm(
        "Ask the main model to review dream reports?",
        default=memory.wake,
    )
    memory.retrieval_mode = prompts.select(
        "Recall retrieval mode",
        [
            ("Auto", "auto"),
            ("Lexical", "lexical"),
            ("Graph", "graph"),
            ("Personalized PageRank", "ppr"),
        ],
        default=memory.retrieval_mode,
    )
    memory.recall_top_k = _ask_int(
        "Recall results (top k)", memory.recall_top_k, 1
    )
    memory.recall_max_chars = _ask_int(
        "Maximum characters in a recall result",
        memory.recall_max_chars,
        200,
    )
    if memory.dream:
        memory.dream_hour_utc = _ask_int(
            "Dream hour (UTC, 0–23)", memory.dream_hour_utc, 0
        )
        memory.dream_rounds = _ask_int(
            "Dream rounds", memory.dream_rounds, 1
        )
    if memory.sidecar:
        binary = prompts.text(
            "Sidecar binary path (blank = managed install under ~/.sora_agent/bin)",
            memory.sidecar_binary or "",
        )
        memory.sidecar_binary = binary.strip() or None
        memory.sidecar_start_timeout = _ask_float(
            "Sidecar startup timeout (seconds)",
            memory.sidecar_start_timeout,
            1.0,
        )


def _configure_files(settings: Settings) -> None:
    files = settings.features.files
    ui.section(
        "Files and images",
        "Download attachments and turn them into markdown appended to the "
        "system prompt. This works even when the main model cannot see images.",
    )
    files.enabled = prompts.confirm(
        "Understand incoming files and images?", default=files.enabled
    )
    if not files.enabled:
        ui.info(
            "Attachments are ignored and the model only receives a note that "
            "they were sent."
        )
        return
    max_mb = _ask_int(
        "Maximum attachment size (MB)",
        max(1, files.max_file_bytes // (1024 * 1024)),
        1,
    )
    files.max_file_bytes = max_mb * 1024 * 1024
    files.max_chars = _ask_int(
        "Maximum characters per attachment", files.max_chars, 500
    )
    files.total_max_chars = _ask_int(
        "Maximum characters for all attachments",
        files.total_max_chars,
        files.max_chars,
    )
    model = prompts.text(
        "Vision model (blank = auto-select or use OCR/metadata)",
        files.vision_model or "",
    )
    files.vision_model = model.strip() or None


def _configure_identity(settings: Settings) -> None:
    ui.section(
        "Identity",
        "Identity stores USER.md and SOUL.md, injects them into the prompt, "
        "and redacts those details from memory writes.",
    )
    enabled = prompts.confirm(
        "Enable persistent identity?", default=settings.features.identity
    )
    settings.features.identity = enabled
    if not enabled:
        ui.warning(
            "Identity is off. Memory redaction against USER.md is unavailable."
        )
        return
    root = prompts.text(
        "Identity directory (blank = ~/.sora_agent/identity)",
        settings.paths.identity_root or "",
    )
    settings.paths.identity_root = root.strip() or None


def _configure_clock_calendar(settings: Settings) -> None:
    ui.section(
        "Clock and calendar",
        "The clock adds the current time to the prompt. The calendar exposes "
        "native event and to-do tools.",
    )
    settings.features.clock = prompts.confirm(
        "Inject the current time into the system prompt?",
        default=settings.features.clock,
    )
    settings.features.calendar = prompts.confirm(
        "Enable the native calendar tools?",
        default=settings.features.calendar,
    )
    if settings.features.calendar:
        db = prompts.text(
            "Calendar database path (blank = ~/.sora_agent/calendar.db)",
            settings.paths.calendar_db or "",
        )
        settings.paths.calendar_db = db.strip() or None


def _configure_sessions(settings: Settings) -> None:
    ui.section(
        "Sessions",
        "Sessions save the exact conversation to disk. With sessions off, "
        "history stays in memory only for the current process; it does not "
        "remove context from the current conversation.",
    )
    enabled = prompts.confirm(
        "Persist sessions across restarts?", default=settings.features.sessions
    )
    settings.features.sessions = enabled
    if enabled:
        root = prompts.text(
            "Sessions directory (blank = ~/.sora_agent/sessions)",
            settings.paths.sessions_root or "",
        )
        settings.paths.sessions_root = root.strip() or None


def _configure_tools_and_limits(settings: Settings) -> None:
    ui.section(
        "Tools and limits",
        "The demo pack is only for trying the approval and quality gates; it "
        "is not a real order system.",
    )
    settings.features.tools = prompts.select(
        "Tool set",
        [("None", "none"), ("Demo order tools", "demo")],
        default=settings.features.tools,
    )
    settings.limits.max_steps = _ask_int(
        "Maximum tool steps per turn", settings.limits.max_steps, 1
    )
    settings.limits.max_turns = _ask_int(
        "Conversation turns kept per session", settings.limits.max_turns, 0
    )
    settings.limits.tool_timeout = _ask_float(
        "Tool timeout (seconds)", settings.limits.tool_timeout, 0.1
    )
    settings.limits.max_tool_result_chars = _ask_int(
        "Maximum tool result characters",
        settings.limits.max_tool_result_chars,
        0,
    )


def _configure_approval(settings: Settings) -> None:
    ui.section(
        "Approval",
        "Approval is a policy gate, not a convenience. Turning it off lets "
        "tools marked consequential execute without asking.",
    )
    if settings.features.approval == "off":
        ui.warning_panel(
            "Approval is OFF",
            [
                "Tools marked as requiring approval will execute immediately.",
                "Every call is still recorded in the audit trail, but nothing",
                "stops it before the side effect happens.",
            ],
        )
    settings.features.approval = prompts.select(
        "Approval policy",
        [
            ("Off — execute without asking", "off"),
            ("Ask — require a human decision", "ask"),
        ],
        default=settings.features.approval,
    )
    if settings.features.approval == "ask":
        settings.limits.approval_timeout = _ask_float(
            "Approval timeout (seconds)",
            settings.limits.approval_timeout,
            1.0,
        )


def _configure_quality(settings: Settings) -> None:
    ui.section(
        "Quality evaluation",
        "The structural quality gate runs after a turn and never changes "
        "execution. With it off, /eval reports that evaluation is disabled.",
    )
    settings.features.quality = prompts.confirm(
        "Enable post-run quality evaluation?", default=settings.features.quality
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _show_advanced_settings(
    settings: Settings, secrets: dict[str, str]
) -> None:
    agent = settings.agent
    ui.kv_table(
        "Agent",
        [
            ("Name", agent.name),
            ("Icon", agent.icon or "(none)"),
            (
                "Personality",
                (
                    f"{_personality_label(agent.personality.base)} · "
                    f"{_WARMTH_LABELS.get(agent.personality.warmth, agent.personality.warmth)}"
                ),
            ),
            ("Temperature", f"{settings.provider.temperature:g}"),
        ],
    )
    ui.kv_table(
        "Model",
        [
            ("Provider", _provider_label(settings.provider.kind)),
            ("Model", settings.provider.model),
            ("Context window", format_tokens(settings.provider.context_window)),
            ("Base URL", settings.provider.base_url or "(provider default)"),
            ("API key", _key_status(settings.provider.kind, secrets)),
            ("Provider timeout", f"{settings.limits.provider_timeout:g}s"),
            ("Provider retries", str(settings.limits.provider_retries)),
        ],
    )
    ui.kv_table(
        "Channel",
        [
            ("Telegram token", _token_status(secrets)),
            (
                "Allowed users",
                ", ".join(settings.channel.allow_from) or "(none)",
            ),
            ("Parse mode", settings.channel.parse_mode or "plain text"),
            ("API base", settings.channel.api_base),
            ("Poll timeout", f"{settings.channel.poll_timeout}s"),
            ("Request timeout", f"{settings.channel.request_timeout:g}s"),
            ("Max message chars", str(settings.channel.max_message_chars)),
        ],
    )
    memory = settings.features.memory
    memory_rows = [("Enabled", "On" if memory.enabled else "Off")]
    if memory.enabled:
        memory_rows.extend(
            [
                ("Sidecar", "On" if memory.sidecar else "Off"),
                (
                    "Auto-download",
                    "On" if memory.auto_download else "Off",
                ),
                ("Dream", "On" if memory.dream else "Off"),
                (
                    "Dream hour",
                    f"{memory.dream_hour_utc:02d}:00 UTC",
                ),
                ("Wake review", "On" if memory.wake else "Off"),
                ("Recall mode", memory.retrieval_mode),
                ("Recall top k", str(memory.recall_top_k)),
                ("Recall max chars", str(memory.recall_max_chars)),
            ]
        )
    ui.kv_table("Memory", memory_rows)
    ui.kv_table(
        "Subsystems",
        [
            ("Identity", "On" if settings.features.identity else "Off"),
            ("Clock", "On" if settings.features.clock else "Off"),
            ("Calendar", "On" if settings.features.calendar else "Off"),
            ("Sessions", "On" if settings.features.sessions else "In-memory only"),
            ("Tools", settings.features.tools),
            (
                "Approval",
                _APPROVAL_LABELS.get(settings.features.approval, settings.features.approval),
            ),
            ("Quality gate", "On" if settings.features.quality else "Off"),
        ],
    )
    file_settings = settings.features.files
    ui.kv_table(
        "Files and images",
        [
            ("Enabled", "On" if file_settings.enabled else "Off"),
            (
                "Max size",
                f"{max(1, file_settings.max_file_bytes // (1024 * 1024))} MB",
            ),
            ("Per-file chars", str(file_settings.max_chars)),
            ("Total chars", str(file_settings.total_max_chars)),
            (
                "Vision model",
                file_settings.vision_model or "auto / OCR fallback",
            ),
        ],
    )
    ui.kv_table(
        "Limits",
        [
            ("Max steps", str(settings.limits.max_steps)),
            ("Max turns", str(settings.limits.max_turns)),
            ("Tool timeout", f"{settings.limits.tool_timeout:g}s"),
            (
                "Approval timeout",
                f"{settings.limits.approval_timeout:g}s",
            ),
            (
                "Max tool result chars",
                str(settings.limits.max_tool_result_chars),
            ),
        ],
    )


def _personality_label(base: str) -> str:
    return _PERSONALITY_LABELS.get(base, base)


def _provider_label(kind: str) -> str:
    info = provider_info(kind)
    return info.label if info is not None else kind


def _key_status(kind: str, secrets: dict[str, str]) -> str:
    info = provider_info(kind)
    if info is None or kind == "mock":
        return "(not required)"
    env_key = os.getenv(info.key_env, "").strip()
    stored = secrets.get(info.key_env, "").strip()
    if env_key:
        return f"{info.key_env} (environment, {_mask(env_key)})"
    if stored:
        return f"{info.key_env} (saved, {_mask(stored)})"
    return "not configured"


def _token_status(secrets: dict[str, str]) -> str:
    env_token = os.getenv("SORA_TELEGRAM_TOKEN", "").strip()
    stored = secrets.get("SORA_TELEGRAM_TOKEN", "").strip()
    if env_token:
        return f"environment ({_mask(env_token)})"
    if stored:
        return f"saved ({_mask(stored)})"
    return "not configured"


def _mask(secret: str) -> str:
    text = str(secret)
    if len(text) <= 8:
        return "•" * max(4, len(text))
    return f"{text[:4]}…{text[-4:]}"


# ---------------------------------------------------------------------------
# Validation and numeric prompts
# ---------------------------------------------------------------------------


def _validate_nonempty(value: str) -> bool | str:
    return True if value.strip() else "a value is required"


def _validate_allow(value: str) -> bool | str:
    text = value.strip()
    if not text:
        return True
    if text == "*":
        return True
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if not item.lstrip("-").isdigit():
            return "use numeric user ids, or * for everyone"
    return True


def _ask_int(
    message: str, default: int, minimum: int, maximum: int | None = None
) -> int:
    while True:
        raw = prompts.text(
            message,
            str(default),
            validate=lambda value: _validate_int(
                value, minimum, maximum
            ),
        )
        try:
            return int(raw)
        except ValueError:
            ui.warning("Enter a whole number.")


def _ask_float(
    message: str,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    while True:
        raw = prompts.text(
            message,
            f"{default:g}",
            validate=lambda value: _validate_float(
                value, minimum, maximum
            ),
        )
        try:
            return float(raw)
        except ValueError:
            ui.warning("Enter a number.")


def _validate_int(
    value: str, minimum: int, maximum: int | None
) -> bool | str:
    try:
        number = int(value)
    except ValueError:
        return "enter a whole number"
    if number < minimum:
        return f"must be at least {minimum}"
    if maximum is not None and number > maximum:
        return f"must be at most {maximum}"
    return True


def _validate_float(
    value: str, minimum: float, maximum: float | None
) -> bool | str:
    try:
        number = float(value)
    except ValueError:
        return "enter a number"
    if number < minimum:
        return f"must be at least {minimum:g}"
    if maximum is not None and number > maximum:
        return f"must be at most {maximum:g}"
    return True


# ---------------------------------------------------------------------------
# Network, finish, and edit-existing
# ---------------------------------------------------------------------------


def _fetch_bot_username(
    token: str, api_base: str = "https://api.telegram.org"
) -> str:
    url = f"{api_base.rstrip('/')}/bot{token}/getMe"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"Telegram rejected the token ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach Telegram: {exc.reason}") from exc
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("description") or "getMe failed"))
    result = payload.get("result") or {}
    username = str(result.get("username") or "").strip()
    if not username:
        raise RuntimeError("Telegram returned no bot username")
    return username


def _finish(settings: Settings, secrets: dict[str, str]) -> None:
    settings.setup.completed_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    ui.blank()
    config_file = save_settings(settings)
    secrets_file = save_secrets(secrets)
    ui.success(f"Configuration saved to {config_file}")
    ui.success(f"Secrets saved to {secrets_file} (mode 0600)")
    if (
        settings.features.memory.enabled
        and settings.features.memory.sidecar
        and prompts.confirm(
            "Download and verify the CortexDB sidecar now?", default=True
        )
    ):
        _prepare_sidecar(settings, secrets)
    _next_steps(settings)


def _prepare_sidecar(settings: Settings, secrets: dict[str, str]) -> None:
    try:
        from ..memory.config import MemoryConfig
        from ..memory.sidecar import SidecarManager
        from ..runtime_setup import openai_compatible_base, resolve_api_key

        memory = settings.features.memory
        config = MemoryConfig.from_settings(
            memory.__dict__,
            root=settings.paths.memory_root,
            wake_model=settings.provider.model if memory.wake else "",
            wake_base_url=(
                openai_compatible_base(settings) or ""
                if memory.wake
                else ""
            ),
            wake_api_key=resolve_api_key(settings, secrets) if memory.wake else "",
        )
        manager = SidecarManager(config)
        try:
            with ui.console.status("Preparing CortexDB sidecar..."):
                client = manager.ensure_running()
                healthy = client.health()
        finally:
            manager.stop()
        if healthy:
            ui.success("CortexDB sidecar is ready.")
        else:
            ui.warning("CortexDB started but its health check failed.")
    except Exception as exc:  # noqa: BLE001 - setup must not fail here
        ui.warning(f"CortexDB sidecar is not ready yet: {exc}")
        ui.info(
            "The gateway will retry on first use. You can also run "
            "`sora-memory doctor` later."
        )


def _next_steps(settings: Settings) -> None:
    ui.blank()
    ui.section("Ready")
    ui.kv_table(
        "Start using Sora",
        [
            ("Terminal chat", "sora chat"),
            ("One-shot task", 'sora run "check the status of PO-1234"'),
            ("Telegram gateway", "sora gateway"),
            ("Health check", "sora doctor"),
            ("Edit settings", "sora setup"),
        ],
    )
    ui.info(
        f"Configuration: {config_path()}   Secrets: {secrets_path()}"
    )


def _edit_existing(settings: Settings, secrets: dict[str, str]) -> None:
    ui.info(
        f"Editing {config_path()} (mode: {settings.setup.mode}). "
        "Values are saved only when you confirm."
    )
    if settings.setup.mode == "advanced":
        _advanced_review(settings, secrets)
        _advanced_final_confirmation(settings)
    else:
        _recommended_review(settings, secrets)
    _finish(settings, secrets)
