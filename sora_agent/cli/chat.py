"""Local terminal chat and one-shot execution."""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from .. import __version__
from ..approval import ApprovalGate, AutoApprovalGate
from ..channels.chat import CHAT_SYSTEM_PROMPT, ChatHandler
from ..channels.types import (
    STREAM_DELTA_KIND,
    STREAM_END_KIND,
    InboundMessage,
    TurnReply,
)
from ..persona import apply_persona, display_name
from ..runtime_setup import (
    approval_mode,
    build_calendar,
    build_identity,
    build_memory,
    build_provider,
    build_registry,
    build_sessions,
    clock_enabled,
    context_window,
    provider_label,
    quality_enabled,
    streaming_enabled,
)
from ..settings import Settings
from . import ui


class ConsoleEmitter:
    """Render cumulative stream snapshots as terminal deltas."""

    def __init__(self, out: Any = None) -> None:
        self.out = out or sys.stdout
        self._sent: dict[str, str] = {}

    def __call__(self, message) -> None:
        kind = message.metadata.get("kind")
        stream_id = str(message.metadata.get("stream_id") or "")
        if kind == STREAM_END_KIND:
            if stream_id in self._sent:
                self.out.write("\n")
                self.out.flush()
                self._sent.pop(stream_id, None)
            return
        if kind != STREAM_DELTA_KIND:
            return
        current = message.content
        previous = self._sent.get(stream_id, "")
        chunk = (
            current.removeprefix(previous)
        )
        if chunk:
            self.out.write(chunk)
            self.out.flush()
        self._sent[stream_id] = current


def build_handler(
    settings: Settings,
    secrets: dict[str, str],
    *,
    sessions: Any = ...,
    streaming: bool | None = None,
    emitter: Any = None,
) -> tuple[ChatHandler, Any]:
    """Build a ChatHandler for local use.  Returns ``(handler, memory)``."""
    identity = build_identity(settings)
    memory = build_memory(settings, identity, secrets)
    provider = build_provider(settings, secrets)
    registry = build_registry(settings)
    if sessions is ...:
        sessions = build_sessions(settings)
    calendar = build_calendar(settings)
    mode = approval_mode(settings)
    gate = AutoApprovalGate() if mode == "off" else ApprovalGate()
    stream = streaming_enabled(settings) if streaming is None else streaming
    prompt = apply_persona(
        CHAT_SYSTEM_PROMPT,
        settings.agent.name,
        settings.agent.personality.base,
        settings.agent.personality.warmth,
    )
    handler = ChatHandler(
        provider,
        registry,
        memory=memory,
        identity=identity,
        approval_gate=gate,
        sessions=sessions,
        quality=None if quality_enabled(settings) else False,
        emitter=emitter if stream else None,
        max_steps=settings.limits.max_steps,
        max_turns=settings.limits.max_turns,
        max_tool_result_chars=settings.limits.max_tool_result_chars,
        max_stream_chars=settings.limits.max_stream_chars,
        tool_timeout=settings.limits.tool_timeout,
        provider_retries=settings.limits.provider_retries,
        clock=clock_enabled(settings),
        calendar=calendar,
        context_window=context_window(settings),
        streaming=stream,
        system_prompt=prompt,
    )
    return handler, memory


def interactive(settings: Settings, secrets: dict[str, str]) -> int:
    emitter = ConsoleEmitter()
    handler, memory = build_handler(settings, secrets, emitter=emitter)
    ui.banner(
        settings.agent.name,
        settings.agent.icon,
        __version__,
        (
            f"{provider_label(settings)} · {settings.provider.model} · "
            f"temperature {settings.provider.temperature:g}"
        ),
    )
    ui.info(
        "Commands: /reset, /status, /eval, /help, /exit. "
        "Ctrl-D also exits."
    )
    session = "cli:local"
    try:
        while True:
            try:
                line = input("You › ")
            except EOFError:
                ui.blank()
                break
            except KeyboardInterrupt:
                ui.blank()
                break
            text = line.strip()
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                break
            if text == "/help":
                _print_help(settings)
                continue
            if text == "/reset":
                handler.reset(session)
                ui.success("Conversation reset.")
                continue
            inbound = InboundMessage(
                channel="cli",
                sender_id="local",
                chat_id=session,
                content=text,
            )
            if text.startswith("/"):
                command_reply = handler.handle_command(inbound)
                if command_reply is not None:
                    ui.console.print(command_reply)
                    continue
            try:
                reply = handler.handle(inbound)
            except KeyboardInterrupt:
                ui.blank()
                ui.warning("Turn interrupted locally; the run may still finish.")
                continue
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                ui.error(f"{type(exc).__name__}: {exc}")
                continue
            _print_reply(reply)
    finally:
        if memory is not None:
            with contextlib.suppress(Exception):
                memory.close()
        ui.info("Bye.")
    return 0


def run_once(settings: Settings, secrets: dict[str, str], task: str) -> int:
    handler, memory = build_handler(settings, secrets, sessions=None)
    try:
        reply = handler.handle(
            InboundMessage(
                channel="cli",
                sender_id="local",
                chat_id="cli:run",
                content=task,
            )
        )
        _print_reply(reply)
    finally:
        if memory is not None:
            with contextlib.suppress(Exception):
                memory.close()
    return 0


def _print_reply(reply: str | TurnReply) -> None:
    if isinstance(reply, TurnReply):
        if not reply.delivered:
            ui.console.print(reply.text)
    else:
        ui.console.print(reply)


def _print_help(settings: Settings) -> None:
    ui.kv_table(
        "Local chat",
        [
            ("/reset", "Forget this conversation"),
            ("/status", "Last run metrics"),
            ("/eval", "Quality report"),
            ("/exit", "Leave the chat"),
            (
                "Agent",
                (
                    f"{display_name(settings.agent)} · "
                    f"{provider_label(settings)} · {settings.provider.model}"
                ),
            ),
        ],
    )
