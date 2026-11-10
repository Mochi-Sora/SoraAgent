"""Tests for token streaming through providers, runner, handler, and manager."""

from __future__ import annotations

import asyncio
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
from sora_agent.providers import ModelTurn, provider_turn
from sora_agent.runner import Runner, RunSpec
from sora_agent.tools import ToolExecutor, ToolRegistry


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
            if should_stop is not None and should_stop():
                break
            on_delta(piece)
        return ModelTurn(text="".join(self.pieces))


class ChunkThenFailProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream_turn(self, messages, tools, on_delta, *, should_stop=None):
        self.calls += 1
        on_delta("partial")
        raise TimeoutError("stream dropped")


class RecordingChannel(BaseChannel):
    name = "fake"

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


def _streamed(messages, kind: str):
    return [
        message
        for message in messages
        if message.metadata.get("kind") == kind
    ]


# ---------------------------------------------------------------------------
# provider streaming
# ---------------------------------------------------------------------------


def test_non_streaming_provider_emits_its_text_once():
    chunks: list[str] = []
    turn = provider_turn(PlainProvider("hi there"), [], [], on_delta=chunks.append)
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
# runner streaming
# ---------------------------------------------------------------------------


def test_runner_forwards_deltas_with_the_step_index():
    chunks: list[tuple[str, int]] = []
    result = Runner(ChunkProvider(), ToolExecutor(ToolRegistry())).run(
        RunSpec(
            messages=[],
            tools=[],
            max_steps=1,
            on_delta=lambda chunk, step: chunks.append((chunk, step)),
        )
    )

    assert chunks == [("Hel", 0), ("lo ", 0), ("world", 0)]
    assert result.metrics.streamed_chars == len("Hello world")


def test_runner_never_retries_a_turn_that_streamed_text():
    provider = ChunkThenFailProvider()
    result = Runner(provider, ToolExecutor(ToolRegistry())).run(
        RunSpec(
            messages=[],
            tools=[],
            max_steps=1,
            provider_retries=3,
            on_delta=lambda chunk, step: None,
        )
    )

    assert provider.calls == 1
    assert result.stop_reason == "provider_error"
    assert result.metrics.provider_retries == 0


# ---------------------------------------------------------------------------
# chat handler streaming
# ---------------------------------------------------------------------------


def test_chat_handler_streams_and_marks_the_reply_delivered():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(ChunkProvider(), max_steps=1, emitter=emitted.append)

    reply = handler.handle(_inbound("hi"))

    assert isinstance(reply, TurnReply)
    assert reply.text == "Hello world"
    assert reply.delivered is True
    deltas = _streamed(emitted, STREAM_DELTA_KIND)
    assert deltas
    assert deltas[-1].content == "Hello world"
    assert _streamed(emitted, STREAM_END_KIND)


def test_chat_handler_streams_the_fallback_text_for_plain_providers():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(
        PlainProvider("plain answer"), max_steps=1, emitter=emitted.append
    )

    reply = handler.handle(_inbound("hi"))

    assert isinstance(reply, TurnReply)
    assert reply.delivered is True
    assert _streamed(emitted, STREAM_DELTA_KIND)[-1].content == "plain answer"


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

    handler.handle(_inbound("hi"))

    deltas = _streamed(emitted, STREAM_DELTA_KIND)
    assert 1 < len(deltas) < 10
    assert deltas[-1].content == "abc" * 10


def test_streaming_can_be_disabled_with_an_emitter_installed():
    emitted: list[OutboundMessage] = []
    handler = ChatHandler(
        PlainProvider("plain answer"),
        max_steps=1,
        emitter=emitted.append,
        streaming=False,
    )

    reply = handler.handle(_inbound("hi"))

    assert reply == "plain answer"
    assert emitted == []


def test_chat_handler_returns_plain_text_without_an_emitter():
    handler = ChatHandler(PlainProvider("plain answer"), max_steps=1)
    assert handler.handle(_inbound("hi")) == "plain answer"


# ---------------------------------------------------------------------------
# manager delivery
# ---------------------------------------------------------------------------


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
                    metadata={"kind": STREAM_DELTA_KIND, "stream_id": "s1"},
                )
            )
            await asyncio.sleep(0.1)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent

    assert asyncio.run(scenario()) == []


def test_manager_delivers_stream_frames_to_a_streaming_channel():
    async def scenario():
        bus = MessageBus()
        channel = StreamingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, lambda inbound: "")
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_outbound(
                OutboundMessage(
                    "fake",
                    "42",
                    "partial",
                    metadata={"kind": STREAM_DELTA_KIND, "stream_id": "s1"},
                )
            )
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent

    sent = asyncio.run(scenario())
    assert [message.content for message in sent] == ["partial"]
