"""Tests for streamed replies, cancellation, and control commands."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from sora_agent.channels.base import BaseChannel
from sora_agent.channels.bus import MessageBus
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.manager import ChannelManager
from sora_agent.channels.types import (
    STREAM_DELTA_KIND,
    STREAM_END_KIND,
    InboundMessage,
    OutboundMessage,
    TurnReply,
)
from sora_agent.providers import ModelTurn, ToolCall, provider_turn
from sora_agent.runner import Runner, RunSpec
from sora_agent.tools import Tool, ToolExecutor, ToolRegistry


class PlainProvider:
    """Only implements ``complete_turn``, like most existing providers."""

    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        return ModelTurn(text=self.text)


class ChunkProvider:
    def __init__(self, pieces=None) -> None:
        self.pieces = pieces or ["Hel", "lo ", "world"]

    def stream_turn(self, messages, tools, on_delta, *, should_stop=None):
        for piece in self.pieces:
            on_delta(piece)
        return ModelTurn(text="".join(self.pieces))


class RecordingChannel(BaseChannel):
    name = "fake"
    supports_streaming = False

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.sent: list[OutboundMessage] = []
        self._stopped: asyncio.Event | None = None

    async def start(self) -> None:
        self._running = True
        self._stopped = asyncio.Event()
        await self._stopped.wait()

    async def stop(self) -> None:
        self._running = False
        if self._stopped is not None:
            self._stopped.set()

    async def send(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)


class StreamingChannel(RecordingChannel):
    supports_streaming = True


def _inbound(text: str) -> InboundMessage:
    return InboundMessage("fake", "7", "42", text)


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# provider streaming
# ---------------------------------------------------------------------------


def test_non_streaming_provider_emits_its_text_once():
    chunks: list[str] = []
    turn = provider_turn(
        PlainProvider("hi there"), [], [], on_delta=chunks.append
    )
    assert turn.text == "hi there"
    assert chunks == ["hi there"]


def test_non_streaming_provider_stays_silent_after_cancel():
    chunks: list[str] = []
    provider_turn(
        PlainProvider("late answer"),
        [],
        [],
        on_delta=chunks.append,
        should_stop=lambda: True,
    )
    assert chunks == []


def test_streaming_provider_controls_the_chunks():
    chunks: list[str] = []
    turn = provider_turn(ChunkProvider(), [], [], on_delta=chunks.append)
    assert turn.text == "Hello world"
    assert chunks == ["Hel", "lo ", "world"]


# ---------------------------------------------------------------------------
# chat handler streaming
# ---------------------------------------------------------------------------


def test_chat_handler_streams_and_marks_the_reply_delivered():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(
        ChunkProvider(), max_steps=1, emitter=emitted.append
    )

    reply = handler.handle(_inbound("hi"))

    assert isinstance(reply, TurnReply)
    assert reply.text == "Hello world"
    assert reply.delivered is True
    deltas = [
        message
        for message in emitted
        if message.metadata.get("kind") == STREAM_DELTA_KIND
    ]
    assert deltas
    assert deltas[-1].content == "Hello world"
    assert any(
        message.metadata.get("kind") == STREAM_END_KIND for message in emitted
    )


def test_chat_handler_streams_the_fallback_text_for_plain_providers():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(
        PlainProvider("plain answer"), max_steps=1, emitter=emitted.append
    )

    reply = handler.handle(_inbound("hi"))

    assert isinstance(reply, TurnReply)
    assert reply.delivered is True
    assert emitted[-2].content == "plain answer"


def test_long_answers_fall_back_to_the_final_message():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(
        ChunkProvider(["x" * 30]),
        max_steps=1,
        emitter=emitted.append,
        max_stream_chars=20,
    )

    reply = handler.handle(_inbound("hi"))

    assert isinstance(reply, TurnReply)
    assert reply.delivered is False


def test_stream_deltas_are_coalesced():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(
        ChunkProvider(["abc"] * 10), max_steps=1, emitter=emitted.append
    )

    reply = handler.handle(_inbound("hi"))

    assert isinstance(reply, TurnReply)
    deltas = [
        message
        for message in emitted
        if message.metadata.get("kind") == STREAM_DELTA_KIND
    ]
    assert 1 < len(deltas) < 10
    assert deltas[-1].content == "abc" * 10


def test_chat_handler_returns_plain_text_without_an_emitter():
    handler = ChatHandler(PlainProvider("plain answer"), max_steps=1)
    assert handler.handle(_inbound("hi")) == "plain answer"


# ---------------------------------------------------------------------------
# runner and handler cancellation
# ---------------------------------------------------------------------------


def test_runner_never_calls_a_cancelled_turn():
    provider = PlainProvider()
    result = Runner(provider, ToolExecutor(ToolRegistry())).run(
        RunSpec(messages=[], tools=[], max_steps=3, should_stop=lambda: True)
    )

    assert result.stop_reason == "cancelled"
    assert provider.calls == 0


def test_runner_cancels_before_the_next_tool_call():
    executed: list[int] = []

    class LoopProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete_turn(self, messages, tools) -> ModelTurn:
            self.calls += 1
            return ModelTurn(
                text="working",
                tool_calls=[
                    ToolCall(id=f"c{self.calls}", name="noop", arguments={})
                ],
            )

    registry = ToolRegistry()
    registry.add(Tool(lambda: executed.append(1) or "ok", "noop", "noop"))
    provider = LoopProvider()

    result = Runner(provider, ToolExecutor(registry)).run(
        RunSpec(
            messages=[],
            tools=registry.all(),
            max_steps=10,
            should_stop=lambda: provider.calls >= 2,
        )
    )

    assert result.stop_reason == "cancelled"
    assert provider.calls == 2
    assert executed == [1]


def test_chat_handler_cancel_stops_a_running_turn():
    class GateProvider:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def complete_turn(self, messages, tools) -> ModelTurn:
            self.started.set()
            self.release.wait(5)
            return ModelTurn(text="late answer")

    provider = GateProvider()
    handler = ChatHandler(provider, max_steps=2)
    result: dict = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("reply", handler.handle(_inbound("hi")))
    )
    worker.start()
    try:
        assert provider.started.wait(2)
        assert handler.cancel("fake:42") is True
        provider.release.set()
    finally:
        worker.join(2)

    assert worker.is_alive() is False
    assert result["reply"] == "Cancelled."
    assert handler.cancel("fake:42") is False


def test_handle_command_parses_control_commands():
    handler = ChatHandler(PlainProvider(), max_steps=1)

    assert handler.handle_command(_inbound("/cancel")) == (
        "Nothing is running right now."
    )
    assert handler.handle_command(_inbound("/cancel@mybot")) == (
        "Nothing is running right now."
    )
    assert handler.handle_command(_inbound("/reset")) == "Conversation reset."
    assert handler.handle_command(_inbound("/unknown")) is None
    assert handler.handle_command(_inbound("hello")) is None


# ---------------------------------------------------------------------------
# manager behaviour
# ---------------------------------------------------------------------------


def test_manager_answers_commands_without_running_the_agent():
    async def scenario():
        bus = MessageBus()
        provider = PlainProvider()
        handler = ChatHandler(provider, max_steps=1)
        channel = RecordingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, handler.handle, commands=handler.handle_command
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("/cancel"))
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return [message.content for message in channel.sent], provider.calls

    contents, calls = asyncio.run(scenario())
    assert contents == ["Nothing is running right now."]
    assert calls == 0


def test_manager_resends_the_final_text_for_a_non_streaming_channel():
    async def scenario():
        bus = MessageBus()
        channel = RecordingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, lambda inbound: TurnReply("final", delivered=True)
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return [message.content for message in channel.sent]

    assert asyncio.run(scenario()) == ["final"]


def test_manager_skips_the_final_text_when_the_channel_streamed_it():
    async def scenario():
        bus = MessageBus()
        channel = StreamingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, lambda inbound: TurnReply("final", delivered=True)
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await asyncio.sleep(0.1)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent

    assert asyncio.run(scenario()) == []


def test_manager_runs_the_shutdown_hook():
    called: list[int] = []

    async def scenario():
        bus = MessageBus()
        channel = RecordingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel],
            bus,
            lambda inbound: "",
            on_shutdown=lambda: called.append(1),
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert called == [1]


def test_manager_drops_stream_frames_for_a_non_streaming_channel():
    async def scenario():
        bus = MessageBus()
        channel = RecordingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, lambda inbound: "")
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_outbound(
                OutboundMessage(
                    "fake",
                    "42",
                    "partial",
                    metadata={
                        "kind": STREAM_DELTA_KIND,
                        "stream_id": "s1",
                    },
                )
            )
            await asyncio.sleep(0.1)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent

    assert asyncio.run(scenario()) == []
