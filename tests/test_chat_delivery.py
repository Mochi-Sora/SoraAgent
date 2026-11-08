"""Tests for completed-reply delivery, typing indicators, and cancellation."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from sora_agent.channels.base import BaseChannel
from sora_agent.channels.bus import MessageBus
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.manager import ChannelManager
from sora_agent.channels.types import InboundMessage, OutboundMessage
from sora_agent.providers import ModelTurn, ToolCall
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


class RecordingChannel(BaseChannel):
    name = "fake"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.sent: list[OutboundMessage] = []
        self.typing: list[InboundMessage] = []
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


class TypingChannel(RecordingChannel):
    async def send_typing(self, inbound: InboundMessage) -> None:
        self.typing.append(inbound)


class BrokenTypingChannel(RecordingChannel):
    async def send_typing(self, inbound: InboundMessage) -> None:
        raise RuntimeError("typing unavailable")


class FlakyChannel(RecordingChannel):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.attempts = 0

    async def send(self, msg: OutboundMessage) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("transient delivery failure")
        await super().send(msg)


class FailingChannel(RecordingChannel):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.attempts = 0

    async def send(self, msg: OutboundMessage) -> None:
        self.attempts += 1
        raise RuntimeError("permanent delivery failure")


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
# handler and runner cancellation
# ---------------------------------------------------------------------------


def test_chat_handler_returns_plain_text():
    handler = ChatHandler(PlainProvider("plain answer"), max_steps=1)
    assert handler.handle(_inbound("hi")) == "plain answer"


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
# manager delivery and typing
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


def test_manager_always_sends_the_final_reply_once():
    async def scenario():
        bus = MessageBus()
        channel = RecordingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, lambda inbound: "final")
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
            await asyncio.sleep(0.05)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return [message.content for message in channel.sent]

    assert asyncio.run(scenario()) == ["final"]


def test_manager_sends_typing_until_the_turn_completes():
    async def scenario():
        bus = MessageBus()
        release = threading.Event()
        started = threading.Event()

        def handler(inbound: InboundMessage) -> str:
            started.set()
            release.wait(5)
            return "answer"

        channel = TypingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, handler, typing_interval=0.01
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await asyncio.to_thread(started.wait, 2)
            await _wait_for(lambda: len(channel.typing) >= 2)
            release.set()
            await _wait_for(lambda: bool(channel.sent))
            count = len(channel.typing)
            await asyncio.sleep(0.05)
            assert len(channel.typing) == count
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel

    channel = asyncio.run(scenario())
    assert all(item.session_key == "fake:42" for item in channel.typing)
    assert channel.sent[-1].content == "answer"


def test_manager_stops_typing_when_the_handler_fails():
    async def scenario():
        bus = MessageBus()
        channel = TypingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel],
            bus,
            lambda inbound: (_ for _ in ()).throw(RuntimeError("boom")),
            typing_interval=0.01,
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
            count = len(channel.typing)
            await asyncio.sleep(0.05)
            assert len(channel.typing) == count
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel

    channel = asyncio.run(scenario())
    assert "couldn't complete" in channel.sent[-1].content


def test_typing_failure_does_not_block_the_reply():
    async def scenario():
        bus = MessageBus()
        channel = BrokenTypingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, lambda inbound: "answer")
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return [message.content for message in channel.sent]

    assert asyncio.run(scenario()) == ["answer"]


def test_transient_delivery_failure_is_retried():
    async def scenario():
        bus = MessageBus()
        channel = FlakyChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, lambda inbound: "answer")
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel

    channel = asyncio.run(scenario())
    assert channel.attempts == 2
    assert [message.content for message in channel.sent] == ["answer"]


def test_permanent_delivery_failure_does_not_crash_the_manager():
    async def scenario():
        bus = MessageBus()
        channel = FailingChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, lambda inbound: "answer")
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: channel.attempts >= 3)
            assert manager._stop is not None
            assert not manager._stop.is_set()
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel

    channel = asyncio.run(scenario())
    assert channel.attempts == 3
    assert channel.sent == []


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
