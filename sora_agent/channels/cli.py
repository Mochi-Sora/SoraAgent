"""Command-line entry point for the channel gateway.

With no config file this behaves exactly like the original environment-only
gateway.  Once ``~/.sora_agent/config.json`` exists, its values become the
defaults; an explicit flag or ``SORA_*`` variable still wins.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import logging
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
from .registry import discover_plugins, load_plugin
from .telegram.config import from_settings as telegram_config_from_settings


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

    channel_kind = (
        args.channel or settings.channel.kind or "telegram"
    ).strip().lower()
    try:
        plugin = load_plugin(channel_kind)
    except Exception as exc:  # noqa: BLE001 - a broken plugin is a CLI error
        print(
            f"sora-gateway: unknown channel {channel_kind!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    try:
        config = build_channel_config(channel_kind, settings, secrets, args)
    except (ValueError, ImportError) as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2
    if not config.allow_from:
        print(
            f"sora-gateway: warning: {channel_kind} allow_from is empty, so "
            "every message will be ignored. Set the channel's ALLOW_FROM "
            "variable, pass --allow <id>, or run `sora setup`.",
            file=sys.stderr,
        )
    try:
        channel_cls = plugin.load_channel_class()
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
    config: Any,
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
            "Run the agent behind a chat channel. Each channel reads its "
            "credentials and access policy from SORA_<CHANNEL>_* variables "
            "unless the flags below override them. A saved `sora setup` "
            "configuration supplies the defaults when no flag or environment "
            "variable is set. Run `sora-gateway --list` to see the installed "
            "channels."
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
        help="comma-separated sender ids; * allows everyone",
    )
    parser.add_argument(
        "--list", action="store_true", help="list discovered channels and exit"
    )
    return parser


def _telegram_config(settings, secrets, args):
    """Backward-compatible wrapper around the Telegram config builder."""
    return telegram_config_from_settings(settings, secrets, args)


def build_channel_config(name: str, settings, secrets, args) -> Any:
    """Load one channel's ``from_settings`` config builder."""
    module = importlib.import_module(f"{__package__}.{name}.config")
    builder = getattr(module, "from_settings", None)
    if not callable(builder):
        raise TypeError(
            f"channel {name!r} does not provide a from_settings config builder"
        )
    return builder(settings, secrets, args)


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
