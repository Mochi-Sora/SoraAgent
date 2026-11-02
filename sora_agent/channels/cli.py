"""Command-line entry point for the channel gateway."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from collections.abc import Sequence
from dataclasses import replace

from ..providers import default_provider
from ..tools import ToolRegistry
from .approval import ApprovalBroker
from .bus import MessageBus
from .chat import ChatHandler
from .manager import ChannelManager
from .registry import discover_plugins, load_channel_class
from .telegram.config import TelegramConfig, split_ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        _print_channels()
        return 0
    if args.channel != "telegram":
        print(
            f"sora-gateway: no configuration is wired for channel "
            f"{args.channel!r} yet",
            file=sys.stderr,
        )
        return 2
    try:
        config = TelegramConfig.from_env()
    except ValueError as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2
    if args.token:
        config = replace(config, token=args.token)
    if args.allow is not None:
        config = replace(config, allow_from=split_ids(args.allow))
    if not config.allow_from:
        print(
            "sora-gateway: warning: allow_from is empty, so every message will "
            "be ignored. Set SORA_TELEGRAM_ALLOW_FROM=<user id> or pass "
            "--allow <user id> (use * to allow everyone).",
            file=sys.stderr,
        )
    try:
        channel_cls = load_channel_class(args.channel)
    except ImportError as exc:
        print(f"sora-gateway: {exc}", file=sys.stderr)
        return 2

    try:
        asyncio.run(
            _serve(
                channel_cls,
                config,
                provider=default_provider(args.provider),
                registry=_registry(args.tools),
                memory=_memory(args.memory),
                identity=_identity(args.identity),
                max_steps=args.max_steps,
                approval_timeout=args.approval_timeout,
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
    registry: ToolRegistry,
    memory,
    identity,
    max_steps: int,
    approval_timeout: float,
) -> None:
    bus = MessageBus()
    approvals = ApprovalBroker(bus, timeout=approval_timeout)
    handler = ChatHandler(
        provider,
        registry,
        memory=memory,
        identity=identity,
        approvals=approvals,
        max_steps=max_steps,
    )
    channel = channel_cls(config, bus)
    manager = ChannelManager(
        [channel], bus, handler.handle, approvals=approvals
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, AttributeError):
            loop.add_signal_handler(sig, manager.request_stop)
    print(
        f"sora-gateway: {channel.name} listening "
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
            "SORA_TELEGRAM_ALLOW_FROM unless the flags below override them."
        ),
    )
    parser.add_argument("--channel", default="telegram", help="channel to run")
    parser.add_argument(
        "--provider",
        default="mock",
        help="model provider: mock, openai, deepseek, or qwen",
    )
    parser.add_argument(
        "--max-steps", type=int, default=8, help="tool steps per turn"
    )
    parser.add_argument(
        "--tools",
        choices=("none", "demo"),
        default="none",
        help="tool set to expose (demo registers the sample order tools)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="enable the CortexDB memory subsystem",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="inject the persistent USER.md/SOUL.md identity",
    )
    parser.add_argument(
        "--approval-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for a chat approval",
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


def _print_channels() -> None:
    plugins = discover_plugins()
    if not plugins:
        print("no channels discovered")
        return
    for name, plugin in sorted(plugins.items()):
        deps = ", ".join(plugin.dependencies) or "none"
        print(f"{name:<12} {plugin.display_name:<14} dependencies: {deps}")


def _registry(kind: str) -> ToolRegistry:
    if kind == "demo":
        from ..tools import registry as demo_tools

        return demo_tools.copy()
    return ToolRegistry()


def _memory(enabled: bool):
    if not enabled:
        return None
    try:
        from ..memory import default_memory

        return default_memory()
    except Exception as exc:  # noqa: BLE001 - memory is optional
        print(f"sora-gateway: memory disabled: {exc}", file=sys.stderr)
        return None


def _identity(enabled: bool):
    if not enabled:
        return None
    try:
        from ..identity import default_identity

        return default_identity()
    except Exception as exc:  # noqa: BLE001 - identity is optional
        print(f"sora-gateway: identity disabled: {exc}", file=sys.stderr)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
