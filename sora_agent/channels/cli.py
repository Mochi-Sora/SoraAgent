"""Command-line entry point for the channel gateway.

With no config file this behaves exactly like the original environment-only
gateway.  Once ``~/.sora_agent/config.json`` exists, its values become the
defaults; an explicit flag or ``SORA_*`` variable still wins.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..approval import AutoApprovalGate
from ..persona import apply_persona, display_name
from ..providers import ProviderConfigError
from ..runner import RunEvent
from ..runtime_setup import (
    approval_mode,
    build_calendar,
    build_file_understander,
    build_identity,
    build_memory,
    build_provider,
    build_registry,
    build_sessions,
    clock_enabled,
    context_window,
    feature_enabled,
    load_runtime,
    memory_enabled,
    quality_enabled,
)
from ..settings import SettingsError
from .approval import ApprovalBroker
from .bus import MessageBus
from .chat import CHAT_SYSTEM_PROMPT, ChatHandler
from .manager import ChannelManager
from .registry import discover_plugins, load_channel_class
from .telegram.config import TelegramConfig, split_ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        _print_channels()
        return 0
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sora_agent.run").setLevel(logging.INFO)

    try:
        settings, secrets, _configured = load_runtime()
    except SettingsError as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2

    channel_kind = args.channel or settings.channel.kind or "telegram"
    if channel_kind != "telegram":
        print(
            f"sora-gateway: no configuration is wired for channel "
            f"{channel_kind!r} yet",
            file=sys.stderr,
        )
        return 2
    try:
        config = _telegram_config(settings, secrets, args)
    except ValueError as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2
    if not config.allow_from:
        print(
            "sora-gateway: warning: allow_from is empty, so every message will "
            "be ignored. Set SORA_TELEGRAM_ALLOW_FROM=<user id>, pass "
            "--allow <user id>, or run `sora setup`.",
            file=sys.stderr,
        )
    try:
        channel_cls = load_channel_class("telegram")
    except ImportError as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2

    try:
        identity = build_identity(
            settings,
            enabled=(
                args.identity
                if args.identity is not None
                else feature_enabled(settings, "identity", "SORA_IDENTITY")
            ),
        )
    except Exception as exc:  # noqa: BLE001 - identity is optional
        print(f"sora-gateway: identity disabled: {exc}", file=sys.stderr)
        identity = None

    memory_on = (
        args.memory
        if args.memory is not None
        else memory_enabled(settings)
    )
    memory = None
    if memory_on:
        try:
            memory = build_memory(settings, identity, secrets)
        except Exception as exc:  # noqa: BLE001 - memory is optional
            print(f"sora-gateway: memory disabled: {exc}", file=sys.stderr)

    sessions = None if args.no_sessions else build_sessions(settings)
    try:
        file_understander = build_file_understander(
            settings, secrets, kind=args.provider, model=args.model
        )
    except Exception as exc:  # noqa: BLE001 - file understanding is optional
        print(f"sora-gateway: file understanding disabled: {exc}", file=sys.stderr)
        file_understander = None
    if file_understander is None:
        config = replace(config, download_attachments=False)

    try:
        calendar = False if args.no_calendar else build_calendar(settings)
        if args.calendar_db and not args.no_calendar:
            from ..calendar import default_calendar

            calendar = default_calendar(args.calendar_db)
    except Exception as exc:  # noqa: BLE001 - calendar is optional
        print(f"sora-gateway: calendar disabled: {exc}", file=sys.stderr)
        calendar = False

    try:
        provider = build_provider(
            settings,
            secrets,
            kind=args.provider,
            model=args.model,
            temperature=args.temperature,
            timeout=args.provider_timeout,
        )
    except (ValueError, ImportError, ProviderConfigError) as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2

    approval = (
        args.approval
        or approval_mode(settings)
    )
    quality = (
        not args.no_quality
        if args.no_quality is not None
        else quality_enabled(settings)
    )
    prompt = apply_persona(
        CHAT_SYSTEM_PROMPT,
        settings.agent.name,
        settings.agent.personality.base,
        settings.agent.personality.warmth,
    )
    if approval == "off":
        print(
            f"{display_name(settings.agent)} gateway: approval prompts are OFF; "
            "consequential tools will execute without confirmation.",
            file=sys.stderr,
        )
    if not quality:
        print(
            "sora-gateway: quality evaluation is off; /eval will not produce "
            "a report.",
            file=sys.stderr,
        )

    try:
        asyncio.run(
            _serve(
                channel_cls,
                config,
                provider=provider,
                registry=_registry(settings, args.tools),
                memory=memory,
                identity=identity,
                sessions=sessions,
                max_steps=(
                    args.max_steps
                    if args.max_steps is not None
                    else settings.limits.max_steps
                ),
                max_turns=(
                    args.max_turns
                    if args.max_turns is not None
                    else settings.limits.max_turns
                ),
                tool_timeout=(
                    args.tool_timeout
                    if args.tool_timeout is not None
                    else settings.limits.tool_timeout
                ),
                provider_retries=(
                    args.provider_retries
                    if args.provider_retries is not None
                    else settings.limits.provider_retries
                ),
                approval_mode=approval,
                approval_timeout=(
                    args.approval_timeout
                    if args.approval_timeout is not None
                    else settings.limits.approval_timeout
                ),
                clock=(
                    not args.no_clock
                    if args.no_clock is not None
                    else clock_enabled(settings)
                ),
                calendar=calendar,
                calendar_db=args.calendar_db,
                file_understander=file_understander,
                context_window=context_window(settings),
                quality=quality,
                system_prompt=prompt,
                max_tool_result_chars=settings.limits.max_tool_result_chars,
                gateway_label=(
                    f"{settings.agent.icon} sora-gateway".strip()
                    if _configured
                    else "sora-gateway"
                ),
            )
        )
    except KeyboardInterrupt:
        return 0
    return 0


async def _serve(
    channel_cls: type,
    config: TelegramConfig,
    *,
    provider,
    registry,
    memory,
    identity,
    sessions,
    max_steps: int,
    max_turns: int,
    tool_timeout: float,
    provider_retries: int,
    approval_mode: str,
    approval_timeout: float,
    clock: bool = True,
    calendar: Any = None,
    calendar_db: Any = None,
    context_window: int | None = None,
    file_understander: Any = None,
    quality: bool = True,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
    max_tool_result_chars: int = 4000,
    gateway_label: str = "sora-gateway",
) -> None:
    bus = MessageBus()
    broker = None
    gate = None
    if approval_mode == "off":
        gate = AutoApprovalGate()
    else:
        broker = ApprovalBroker(bus, timeout=approval_timeout)
    loop = asyncio.get_running_loop()

    handler = ChatHandler(
        provider,
        registry,
        memory=memory,
        identity=identity,
        approvals=broker,
        approval_gate=gate,
        sessions=sessions,
        on_event=_log_run_event,
        max_steps=max_steps,
        max_turns=max_turns,
        max_tool_result_chars=max_tool_result_chars,
        tool_timeout=tool_timeout,
        provider_retries=provider_retries,
        clock=clock,
        calendar=calendar,
        calendar_db=calendar_db,
        context_window=context_window,
        file_understander=file_understander,
        quality=None if quality else False,
        system_prompt=system_prompt,
    )
    channel = channel_cls(config, bus)
    manager = ChannelManager(
        [channel],
        bus,
        handler.handle,
        approvals=broker,
        commands=handler.handle_command,
        on_shutdown=handler.cancel_all,
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, AttributeError):
            loop.add_signal_handler(sig, manager.request_stop)
    print(
        f"{gateway_label}: {channel.name} listening "
        f"(allow_from={'*' if '*' in config.allow_from else len(config.allow_from)})",
        file=sys.stderr,
    )
    try:
        await manager.serve()
    finally:
        if memory is not None:
            with contextlib.suppress(Exception):
                memory.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sora-gateway",
        description=(
            "Run the agent behind a chat channel. Telegram reads its token and "
            "allow list from SORA_TELEGRAM_TOKEN and "
            "SORA_TELEGRAM_ALLOW_FROM unless the flags below override them. "
            "A saved `sora setup` configuration supplies the defaults when no "
            "flag or environment variable is set."
        ),
    )
    parser.add_argument("--channel", default=None, help="channel to run")
    parser.add_argument(
        "--provider",
        default=None,
        help="model provider: mock, openai, deepseek, qwen, or custom",
    )
    parser.add_argument(
        "--model", default=None, help="model id (overrides provider config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature (overrides config and SORA_TEMPERATURE)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="tool steps per turn"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="conversation turns kept per session",
    )
    parser.add_argument(
        "--no-sessions",
        action="store_true",
        default=None,
        help="keep conversation history in memory only",
    )
    parser.add_argument(
        "--provider-timeout",
        type=float,
        default=None,
        help="seconds before a provider request is abandoned",
    )
    parser.add_argument(
        "--provider-retries",
        type=int,
        default=None,
        help="retries for transient provider failures",
    )
    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=None,
        help="seconds before an isolated tool process is killed",
    )
    parser.add_argument(
        "--tools",
        choices=("none", "demo"),
        default=None,
        help="tool set to expose (demo registers the sample order tools)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=None,
        help="enable the CortexDB memory subsystem",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        default=None,
        help="inject the persistent USER.md/SOUL.md identity",
    )
    parser.add_argument(
        "--approval",
        choices=("off", "ask"),
        default=None,
        help="approval policy: off (auto-approve) or ask (default)",
    )
    parser.add_argument(
        "--approval-timeout",
        type=float,
        default=None,
        help="seconds to wait for a chat approval",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        default=None,
        help="disable post-run quality evaluation",
    )
    parser.add_argument(
        "--no-clock",
        action="store_true",
        default=None,
        help="omit the current time section from the system prompt",
    )
    parser.add_argument(
        "--no-calendar",
        action="store_true",
        default=None,
        help="disable the native calendar tools",
    )
    parser.add_argument(
        "--calendar-db",
        default=None,
        help="path to the calendar SQLite database "
        "(default: ~/.sora_agent/calendar.db)",
    )
    parser.add_argument("--token", default=None, help="Telegram bot token")
    parser.add_argument(
        "--allow",
        default=None,
        help="comma-separated Telegram user ids; * allows everyone",
    )
    parser.add_argument(
        "--list", action="store_true", help="list discovered channels and exit"
    )
    return parser


def _telegram_config(settings, secrets, args) -> TelegramConfig:
    token = (
        args.token
        or os.getenv("SORA_TELEGRAM_TOKEN", "").strip()
        or secrets.get("SORA_TELEGRAM_TOKEN", "").strip()
    )
    if not token:
        raise ValueError(
            "SORA_TELEGRAM_TOKEN is not set; create a bot with @BotFather and "
            "run `sora setup`, export the token, or pass --token"
        )

    allow_raw = os.getenv("SORA_TELEGRAM_ALLOW_FROM")
    if args.allow is not None:
        allow_from = split_ids(args.allow)
    elif allow_raw is not None:
        allow_from = split_ids(allow_raw)
    else:
        allow_from = tuple(settings.channel.allow_from)

    parse_raw = os.getenv("SORA_TELEGRAM_PARSE_MODE")
    parse_mode = (
        (parse_raw.strip() or None)
        if parse_raw is not None
        else settings.channel.parse_mode
    )

    def env_or(name: str, fallback):
        raw = os.getenv(name)
        return fallback if raw is None or not raw.strip() else raw.strip()

    return TelegramConfig(
        token=token,
        allow_from=allow_from,
        api_base=env_or(
            "SORA_TELEGRAM_API_BASE", settings.channel.api_base
        ),
        poll_timeout=_as_int(
            env_or("SORA_TELEGRAM_POLL_TIMEOUT", settings.channel.poll_timeout),
            settings.channel.poll_timeout,
        ),
        request_timeout=_as_float(
            env_or(
                "SORA_TELEGRAM_REQUEST_TIMEOUT",
                settings.channel.request_timeout,
            ),
            settings.channel.request_timeout,
        ),
        parse_mode=parse_mode,
        max_message_chars=_as_int(
            env_or(
                "SORA_TELEGRAM_MAX_CHARS",
                settings.channel.max_message_chars,
            ),
            settings.channel.max_message_chars,
        ),
    )


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _log_run_event(event: RunEvent) -> None:
    logging.getLogger("sora_agent.run").info(
        "run %s step=%s name=%s ok=%s duration=%.3fs %s",
        event.kind,
        event.step,
        event.name,
        event.ok,
        event.duration,
        event.detail,
    )


def _print_channels() -> None:
    plugins = discover_plugins()
    if not plugins:
        print("no channels discovered")
        return
    for name, plugin in sorted(plugins.items()):
        deps = ", ".join(plugin.dependencies) or "none"
        print(f"{name:<12} {plugin.display_name:<14} dependencies: {deps}")


def _registry(settings, kind: str | None):
    if kind is not None:
        from ..tools import ToolRegistry

        if kind == "demo":
            from ..tools import registry as demo_tools

            return demo_tools.copy()
        return ToolRegistry()
    return build_registry(settings)


if __name__ == "__main__":
    raise SystemExit(main())
