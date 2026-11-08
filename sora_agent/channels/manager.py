"""Start channels, pump inbound to the handler, deliver outbound replies."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Union

from .approval import APPROVAL_CALLBACK_KIND, ApprovalBroker
from .base import BaseChannel
from .bus import MessageBus
from .types import InboundMessage, OutboundMessage

Handler = Callable[[InboundMessage], Union[str, Awaitable[str]]]

_MAX_SEND_ATTEMPTS = 3
_DEFAULT_TYPING_INTERVAL = 4.0
_DEFAULT_MAX_PENDING_TURNS = 32
_DEFAULT_MAX_TURNS_PER_MINUTE = 20
_RATE_WINDOW_SECONDS = 60.0


class ChannelManager:
    """Supervise channels for one process.

    A channel's ``start()`` is a long-running task.  If it returns or raises, the
    manager shuts the whole gateway down: a partially connected agent that still
    accepts messages is worse than one that stops loudly.
    """

    def __init__(
        self,
        channels: Sequence[BaseChannel],
        bus: MessageBus,
        handler: Handler,
        *,
        approvals: ApprovalBroker | None = None,
        commands: Callable[[InboundMessage], str | None] | None = None,
        on_shutdown: Callable[[], object] | None = None,
        typing_interval: float = _DEFAULT_TYPING_INTERVAL,
        max_pending_turns: int = _DEFAULT_MAX_PENDING_TURNS,
        max_turns_per_minute: int = _DEFAULT_MAX_TURNS_PER_MINUTE,
        logger: logging.Logger | None = None,
    ) -> None:
        if not channels:
            raise ValueError("at least one channel is required")
        self.channels = list(channels)
        self.bus = bus
        self.handler = handler
        self.approvals = approvals
        self.commands = commands
        self.on_shutdown = on_shutdown
        self.typing_interval = max(0.01, float(typing_interval))
        self.max_pending_turns = max(1, int(max_pending_turns))
        self.max_turns_per_minute = max(1, int(max_turns_per_minute))
        self.log = logger or logging.getLogger(__name__)
        self._by_name: dict[str, BaseChannel] = {}
        for channel in self.channels:
            if channel.name in self._by_name:
                raise ValueError(f"duplicate channel name: {channel.name}")
            self._by_name[channel.name] = channel
        self._stop: asyncio.Event | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._turns: set[asyncio.Task[Any]] = set()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._turn_times: dict[str, list[float]] = {}
        self._rate_warned: dict[str, float] = {}
        self._shutdown = False

    @property
    def names(self) -> list[str]:
        return [channel.name for channel in self.channels]

    def request_stop(self) -> None:
        """Ask :meth:`serve` to wind down."""
        if self._stop is not None:
            self._stop.set()

    async def serve(self) -> None:
        """Run until :meth:`request_stop` or a channel dies."""
        self._stop = asyncio.Event()
        if self.approvals is not None:
            self.approvals.bind(asyncio.get_running_loop())
        self._tasks = [
            asyncio.create_task(self._run_channel(channel), name=f"ch:{channel.name}")
            for channel in self.channels
        ]
        self._tasks.append(asyncio.create_task(self._dispatch(), name="ch:dispatch"))
        self._tasks.append(asyncio.create_task(self._deliver(), name="ch:deliver"))
        try:
            await self._stop.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop channels and drain the tasks.  Safe to call twice."""
        if self._shutdown:
            return
        self._shutdown = True
        self.request_stop()
        if self.approvals is not None:
            # Release worker threads blocked on an approval, or the executor
            # shutdown would wait out the whole timeout.
            self.approvals.cancel_all()
        if self.on_shutdown is not None:
            try:
                self.on_shutdown()
            except Exception:
                self.log.exception("shutdown hook failed")
        for channel in self.channels:
            try:
                await channel.stop()
            except Exception:
                self.log.exception("channel %s failed to stop", channel.name)
        pending = [*self._tasks, *self._turns]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks = []
        self._turns.clear()

    async def _run_channel(self, channel: BaseChannel) -> None:
        self.log.info("channel %s starting", channel.name)
        try:
            await channel.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception("channel %s failed", channel.name)
        else:
            self.log.warning("channel %s stopped unexpectedly", channel.name)
        finally:
            self.request_stop()

    async def _dispatch(self) -> None:
        while True:
            inbound = await self.bus.next_inbound()
            if inbound.metadata.get("kind") == APPROVAL_CALLBACK_KIND:
                # A button press is never chat input, even when it is stale.
                consumed = False
                if self.approvals is not None:
                    consumed = self.approvals.consume(inbound)
                channel = self._by_name.get(inbound.channel)
                if channel is not None:
                    try:
                        await channel.acknowledge_callback(
                            inbound, consumed
                        )
                    except Exception:
                        self.log.debug(
                            "callback acknowledgement failed", exc_info=True
                        )
                continue
            if self.approvals is not None and self.approvals.consume(inbound):
                self.log.debug(
                    "resolved pending approval for %s", inbound.session_key
                )
                continue
            if self.commands is not None:
                command = self._command_reply(inbound)
                if command is not None:
                    await self._publish_reply(inbound, command)
                    continue
            if len(self._turns) >= self.max_pending_turns:
                self.log.warning(
                    "turn queue full (%d); dropping message from %s",
                    self.max_pending_turns,
                    inbound.session_key,
                )
                continue
            if not self._allow_turn(inbound.session_key):
                now = time.monotonic()
                if now - self._rate_warned.get(inbound.session_key, 0.0) >= (
                    _RATE_WINDOW_SECONDS
                ):
                    self._rate_warned[inbound.session_key] = now
                    await self._publish_reply(
                        inbound,
                        "I'm receiving messages too quickly. Please wait a "
                        "moment and try again.",
                    )
                continue
            # Turns run as separate tasks: a turn blocked on a human approval
            # must not stop the dispatcher from reading the reply.
            task = asyncio.create_task(
                self._run_turn(inbound), name=f"turn:{inbound.session_key}"
            )
            self._turns.add(task)
            task.add_done_callback(self._turns.discard)

    def _allow_turn(self, session_key: str) -> bool:
        now = time.monotonic()
        times = self._turn_times.setdefault(session_key, [])
        times[:] = [
            value
            for value in times
            if now - value < _RATE_WINDOW_SECONDS
        ]
        if len(times) >= self.max_turns_per_minute:
            return False
        times.append(now)
        if len(self._turn_times) > 1024:
            stale = [
                key
                for key, values in self._turn_times.items()
                if not values or now - values[-1] >= _RATE_WINDOW_SECONDS
            ]
            for key in stale:
                self._turn_times.pop(key, None)
                self._rate_warned.pop(key, None)
        return True

    async def _run_turn(self, inbound: InboundMessage) -> None:
        lock = self._session_locks.setdefault(inbound.session_key, asyncio.Lock())
        async with lock:
            channel = self._by_name.get(inbound.channel)
            typing = (
                asyncio.create_task(self._typing_loop(channel, inbound))
                if channel is not None
                else None
            )
            try:
                try:
                    result = await self._invoke(inbound)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.log.exception(
                        "handler failed for %s", inbound.session_key
                    )
                    result = "Sorry — I couldn't complete that request."
            finally:
                if typing is not None:
                    typing.cancel()
                    await asyncio.gather(typing, return_exceptions=True)
            text = _reply_text(result)
            if not text:
                return
            await self._publish_reply(inbound, text)

    async def _typing_loop(
        self, channel: BaseChannel, inbound: InboundMessage
    ) -> None:
        while True:
            try:
                await channel.send_typing(inbound)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - typing is best-effort
                self.log.debug(
                    "typing indicator failed for %s: %s",
                    inbound.session_key,
                    exc,
                )
                return
            await asyncio.sleep(self.typing_interval)

    def _command_reply(self, inbound: InboundMessage) -> str | None:
        handler = self.commands
        if handler is None:
            return None
        try:
            return handler(inbound)
        except Exception:
            self.log.exception(
                "command handler failed for %s", inbound.session_key
            )
            return None

    async def _publish_reply(self, inbound: InboundMessage, text: str) -> None:
        if not text:
            return
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=text,
                reply_to=_reply_target(inbound),
                metadata=dict(inbound.metadata),
            )
        )

    async def _deliver(self) -> None:
        while True:
            message = await self.bus.next_outbound()
            channel = self._by_name.get(message.channel)
            if channel is None:
                self.log.error("no channel named %r to deliver to", message.channel)
                continue
            await self._send(channel, message)

    async def _send(self, channel: BaseChannel, message: OutboundMessage) -> None:
        for attempt in range(1, _MAX_SEND_ATTEMPTS + 1):
            try:
                await channel.send(message)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry policy lives here
                if attempt == _MAX_SEND_ATTEMPTS:
                    self.log.error(
                        "delivery via %s failed after %d attempts: %s",
                        channel.name,
                        attempt,
                        exc,
                    )
                    return
                self.log.warning(
                    "delivery via %s failed (attempt %d): %s",
                    channel.name,
                    attempt,
                    exc,
                )
                await asyncio.sleep(0.2 * attempt)

    async def _invoke(self, inbound: InboundMessage) -> Any:
        handler = self.handler
        is_async = inspect.iscoroutinefunction(handler)
        if not is_async and not inspect.isroutine(handler):
            # A callable object with an async __call__ still needs awaiting.
            is_async = inspect.iscoroutinefunction(type(handler).__call__)
        if is_async:
            value = handler(inbound)
            if inspect.isawaitable(value):
                value = await value
        else:
            # A synchronous handler (the common case: the agent loop) must not
            # block the event loop while it runs.
            value = await asyncio.to_thread(handler, inbound)
            if inspect.isawaitable(value):
                value = await value
        return value


def _reply_text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _reply_target(inbound: InboundMessage) -> str | None:
    value = inbound.metadata.get("message_id")
    return None if value is None else str(value)
