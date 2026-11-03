"""Tests for the channel core: bus, base gate, registry, manager, chat."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sora_agent.channels.approval import ApprovalBroker
from sora_agent.channels.base import BaseChannel
from sora_agent.channels.bus import MessageBus
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.cli import main as cli_main
from sora_agent.channels.manager import ChannelManager
from sora_agent.channels.manifest import ChannelPlugin
from sora_agent.channels.registry import discover_plugins, load_plugin
from sora_agent.channels.types import InboundMessage, OutboundMessage
from sora_agent.eval import EvalReport, QualityGate
from sora_agent.providers import ModelTurn, ToolCall
from sora_agent.tool_search import FIND_TOOLS_NAME
from sora_agent.tools import Tool, ToolRegistry


class FakeChannel(BaseChannel):
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


class EchoProvider:
    def __init__(self) -> None:
        self.calls: list[list] = []
        self.seen_tools: list[list[str]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        self.seen_tools.append([tool.name for tool in tools])
        last = [message for message in messages if message.role == "user"][-1]
        return ModelTurn(text=f"echo:{last.content}")


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.01)


def _inbound(content: str) -> InboundMessage:
    return InboundMessage("fake", "9", "42", content)


# ---------------------------------------------------------------------------
# allow list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("allow", "sender", "expected"),
    [
        ((), "1", False),
        (("*",), "1", True),
        (("1", "2"), "2", True),
        (("1",), "2", False),
        ("1, 2", "2", True),
    ],
)
def test_allow_list(allow, sender, expected):
    channel = FakeChannel(SimpleNamespace(allow_from=allow), MessageBus())
    assert channel.is_allowed(sender) is expected


def test_handle_message_publishes_inbound():
    async def scenario():
        bus = MessageBus()
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        accepted = await channel._handle_message(
            "9", "42", "hi", metadata={"message_id": 7}
        )
        message = await bus.next_inbound()
        return accepted, message

    accepted, message = asyncio.run(scenario())
    assert accepted is True
    assert message.channel == "fake"
    assert message.session_key == "fake:42"
    assert message.metadata == {"message_id": 7}


def test_handle_message_denies_and_publishes_nothing():
    async def scenario():
        bus = MessageBus()
        channel = FakeChannel(SimpleNamespace(allow_from=("7",)), bus)
        accepted = await channel._handle_message("9", "42", "hi")
        return accepted, bus.inbound.qsize()

    accepted, pending = asyncio.run(scenario())
    assert accepted is False
    assert pending == 0


def test_session_key_override_wins():
    message = InboundMessage("telegram", "9", "42", "hi", session_key_override="t:1:2")
    assert message.session_key == "t:1:2"


# ---------------------------------------------------------------------------
# manifests and discovery
# ---------------------------------------------------------------------------


def test_manifest_rejects_bad_name():
    with pytest.raises(ValueError):
        ChannelPlugin(name="9bad", display_name="x", runtime="m:C")


def test_manifest_rejects_bad_runtime():
    with pytest.raises(ValueError):
        ChannelPlugin(name="ok", display_name="x", runtime="nocolon")


def test_manifest_rejects_list_dependencies():
    with pytest.raises(TypeError):
        ChannelPlugin(
            name="ok",
            display_name="x",
            runtime="m:C",
            dependencies=["pkg"],  # type: ignore[arg-type]
        )


def test_discovery_is_dependency_free():
    sys.modules.pop("sora_agent.channels.telegram.runtime", None)
    plugins = discover_plugins()
    assert "telegram" in plugins
    assert plugins["telegram"].runtime.endswith(":TelegramChannel")
    assert "sora_agent.channels.telegram.runtime" not in sys.modules


def test_enabled_filter_skips_other_packages():
    assert discover_plugins({"not_a_channel"}) == {}


def test_load_channel_class_resolves_runtime():
    channel_cls = load_plugin("telegram").load_channel_class()
    assert channel_cls.name == "telegram"


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------


def test_manager_round_trip():
    async def scenario():
        bus = MessageBus()
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, lambda inbound: f"echo {inbound.content}"
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(
                InboundMessage("fake", "9", "42", "hi", metadata={"message_id": 5})
            )
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent[0]

    sent = asyncio.run(scenario())
    assert sent.content == "echo hi"
    assert sent.chat_id == "42"
    assert sent.reply_to == "5"


def test_manager_supports_async_handlers():
    async def handler(inbound: InboundMessage) -> str:
        return f"async {inbound.content}"

    async def scenario():
        bus = MessageBus()
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, handler)
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent[0].content

    assert asyncio.run(scenario()) == "async hi"


def test_manager_serializes_turns_for_the_same_session():
    """Two messages from one chat must not run their agent turns at once."""
    timeline: list[tuple[str, str]] = []
    guard = threading.Lock()

    def handler(inbound: InboundMessage) -> str:
        with guard:
            timeline.append(("start", inbound.content))
        time.sleep(0.05)
        with guard:
            timeline.append(("end", inbound.content))
        return f"echo {inbound.content}"

    async def scenario():
        bus = MessageBus()
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, handler)
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("one"))
            await bus.publish_inbound(_inbound("two"))
            await _wait_for(lambda: len(channel.sent) >= 2)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return list(timeline), [message.content for message in channel.sent]

    timeline, contents = asyncio.run(scenario())
    assert contents == ["echo one", "echo two"]
    assert [kind for kind, _text in timeline] == ["start", "end", "start", "end"]


def test_manager_reports_handler_failure_to_the_user():
    def boom(_inbound: InboundMessage) -> str:
        raise RuntimeError("model exploded")

    async def scenario():
        bus = MessageBus()
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager([channel], bus, boom)
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("hi"))
            await _wait_for(lambda: bool(channel.sent))
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 2)
        return channel.sent[0].content

    assert "couldn't complete" in asyncio.run(scenario())


def test_manager_rejects_duplicate_channel_names():
    bus = MessageBus()
    one = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
    two = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
    with pytest.raises(ValueError, match="duplicate"):
        ChannelManager([one, two], bus, lambda inbound: "")


# ---------------------------------------------------------------------------
# chat handler
# ---------------------------------------------------------------------------


def test_chat_handler_keeps_history():
    provider = EchoProvider()
    handler = ChatHandler(provider, max_steps=1)

    assert handler.handle(_inbound("one")) == "echo:one"
    assert handler.handle(_inbound("two")) == "echo:two"

    roles = [message.role for message in provider.calls[-1]]
    assert roles == ["system", "system", "user", "assistant", "user"]


def test_chat_handler_reset_forgets_the_session():
    provider = EchoProvider()
    handler = ChatHandler(provider, max_steps=1)
    handler.handle(_inbound("one"))
    handler.reset("fake:42")
    handler.handle(_inbound("two"))

    roles = [message.role for message in provider.calls[-1]]
    assert roles == ["system", "system", "user"]


def test_chat_handler_trims_old_turns():
    provider = EchoProvider()
    handler = ChatHandler(provider, max_steps=1, max_turns=1)
    for text in ("a", "b", "c"):
        handler.handle(_inbound(text))

    roles = [message.role for message in provider.calls[-1]]
    assert roles == ["system", "system", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# memory and approval wiring
# ---------------------------------------------------------------------------


class FakeMemoryTool:
    name = "memory_recall"

    def __init__(self) -> None:
        self.queries: list[str] = []

    @property
    def schema(self) -> dict:
        return {
            "name": "memory_recall",
            "description": "Recall relevant memory.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }

    def call(self, query: str = "", **_: object) -> str:
        self.queries.append(query)
        return "remembered: the user likes tea"


class FakeMemory:
    def __init__(self) -> None:
        self.tool_obj = FakeMemoryTool()
        self.before: list[tuple[str, str | None]] = []
        self.after: list = []
        self.identity = None

    def set_identity(self, identity) -> None:
        self.identity = identity

    def tool(self):
        return self.tool_obj

    def before_run(self, task, state=None) -> None:
        self.before.append((task, getattr(state, "session_id", None)))

    def after_run(self, result, state=None) -> None:
        self.after.append(result)


class MemoryProvider:
    def __init__(self) -> None:
        self.calls: list[list] = []
        self.seen_tools: list[list[str]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        self.seen_tools.append([tool.name for tool in tools])
        if not any(message.role == "tool" for message in messages):
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="m1",
                        name="memory_recall",
                        arguments={"query": "hi"},
                    )
                ]
            )
        return ModelTurn(text="answered")


def test_chat_handler_recalls_memory_first():
    memory = FakeMemory()
    provider = MemoryProvider()
    handler = ChatHandler(provider, memory=memory, max_steps=3)

    assert handler.handle(_inbound("hi")) == "answered"
    assert memory.before == [("hi", "fake:42")]
    assert len(memory.after) == 1
    assert memory.after[0].final_answer == "answered"
    assert memory.tool_obj.queries == ["hi"]
    assert provider.seen_tools[0] == ["memory_recall", FIND_TOOLS_NAME]
    assert "## Memory" in provider.calls[0][0].content
    assert "## Tools" in provider.calls[0][0].content


def test_chat_handler_skips_memory_policy_without_memory():
    provider = EchoProvider()
    handler = ChatHandler(provider, max_steps=1)
    assert handler.handle(_inbound("hi")) == "echo:hi"
    assert "## Memory" not in provider.calls[0][0].content
    assert provider.seen_tools[0] == [FIND_TOOLS_NAME]


def test_chat_handler_survives_a_broken_memory_hook():
    class BrokenMemory(FakeMemory):
        def after_run(self, result, state=None) -> None:
            raise RuntimeError("memory is down")

    memory = BrokenMemory()
    handler = ChatHandler(MemoryProvider(), memory=memory, max_steps=3)
    assert handler.handle(_inbound("hi")) == "answered"
    assert memory.tool_obj.queries == ["hi"]


class DiscoveryProvider:
    def __init__(self) -> None:
        self.seen_tools: list[list[str]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.seen_tools.append([tool.name for tool in tools])
        tool_names = [
            message.name for message in messages if message.role == "tool"
        ]
        if not tool_names:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "expedite a late purchase order"},
                    )
                ]
            )
        if "expedite_order" not in tool_names:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="expedite_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            )
        return ModelTurn(text="done")


def test_chat_handler_uses_find_tools_for_discovery():
    executed: list[str] = []
    registry = ToolRegistry()
    registry.add(
        Tool(
            lambda order_id: executed.append(order_id) or "expedited",
            "expedite_order",
            "Expedite a delayed purchase order.",
        )
    )
    provider = DiscoveryProvider()
    handler = ChatHandler(provider, registry, max_steps=4)

    assert handler.handle(_inbound("expedite PO-1")) == "done"
    assert provider.seen_tools[0] == [FIND_TOOLS_NAME]
    assert "expedite_order" in provider.seen_tools[1]
    assert executed == ["PO-1"]


def test_chat_handler_reports_without_blocking_execution():
    executed: list[str] = []
    registry = ToolRegistry()
    registry.add(
        Tool(
            lambda: executed.append("ran") or "done",
            "side_effect",
            "Perform a side effect.",
        )
    )

    class ActionProvider:
        def complete_turn(self, messages, tools) -> ModelTurn:
            if any(message.role == "tool" for message in messages):
                return ModelTurn(text="finished")
            return ModelTurn(
                text="Trying.",
                tool_calls=[ToolCall(id="c1", name="side_effect", arguments={})],
            )

    gate = QualityGate(
        lambda trajectory: EvalReport(False, 0.0, ["advisory only"])
    )
    handler = ChatHandler(
        ActionProvider(), registry, quality=gate, max_steps=2
    )

    assert handler.handle(_inbound("do it")) == "finished"
    assert executed == ["ran"]
    report = handler.last_report("fake:42")
    assert report is not None
    assert report.passed is False


class ApprovalProvider:
    def complete_turn(self, messages, tools) -> ModelTurn:
        if any(message.role == "tool" for message in messages):
            return ModelTurn(text="done")
        return ModelTurn(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="expedite_order",
                    arguments={"order_id": "PO-1"},
                )
            ]
        )


class WrappedProvider:
    __sora_identity__ = True

    def __init__(self, inner) -> None:
        self.inner = inner

    def complete_turn(self, messages, tools) -> ModelTurn:
        return self.inner.complete_turn(messages, tools)


class FakeIdentity:
    def __init__(self) -> None:
        self.wrapped: list = []

    def wrap_provider(self, provider):
        self.wrapped.append(provider)
        return WrappedProvider(provider)


def test_chat_handler_wraps_the_provider_with_identity():
    provider = EchoProvider()
    identity = FakeIdentity()
    handler = ChatHandler(provider, identity=identity, max_steps=1, clock=False)

    assert isinstance(handler.provider, WrappedProvider)
    assert handler.provider.inner is provider
    assert identity.wrapped == [provider]
    assert handler.handle(_inbound("hi")) == "echo:hi"


def test_chat_handler_does_not_double_wrap_identity():
    already = WrappedProvider(EchoProvider())
    identity = FakeIdentity()
    handler = ChatHandler(already, identity=identity, max_steps=1, clock=False)

    assert handler.provider is already
    assert identity.wrapped == []


def test_chat_handler_attaches_identity_to_memory():
    memory = FakeMemory()
    identity = FakeIdentity()
    ChatHandler(EchoProvider(), memory=memory, identity=identity, max_steps=1)

    assert memory.identity is identity


def test_chat_handler_survives_a_failing_identity_wrapper():
    class BrokenIdentity(FakeIdentity):
        def wrap_provider(self, provider):
            raise RuntimeError("identity is down")

    provider = EchoProvider()
    handler = ChatHandler(provider, identity=BrokenIdentity(), max_steps=1, clock=False)

    assert handler.provider is provider
    assert handler.handle(_inbound("hi")) == "echo:hi"


def test_chat_approval_round_trip():
    executed: list[str] = []

    def expedite(order_id: str) -> str:
        executed.append(order_id)
        return "expedited"

    registry = ToolRegistry()
    registry.add(
        Tool(expedite, "expedite_order", "Expedite an order.", needs_approval=True)
    )

    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        handler = ChatHandler(
            ApprovalProvider(), registry, approvals=broker, max_steps=2
        )
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, handler.handle, approvals=broker
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(InboundMessage("fake", "7", "42", "go"))
            await _wait_for(
                lambda: any("Approval required" in m.content for m in channel.sent)
            )
            await bus.publish_inbound(InboundMessage("fake", "7", "42", "yes"))
            await _wait_for(
                lambda: any(m.content == "done" for m in channel.sent)
            )
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return [m.content for m in channel.sent]

    contents = asyncio.run(scenario())
    assert any("Approval required" in content for content in contents)
    assert "done" in contents
    assert executed == ["PO-1"]


def test_chat_approval_denial_blocks_the_tool():
    executed: list[str] = []

    def expedite(order_id: str) -> str:
        executed.append(order_id)
        return "expedited"

    registry = ToolRegistry()
    registry.add(
        Tool(expedite, "expedite_order", "Expedite an order.", needs_approval=True)
    )

    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        handler = ChatHandler(
            ApprovalProvider(), registry, approvals=broker, max_steps=2
        )
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, handler.handle, approvals=broker
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(InboundMessage("fake", "7", "42", "go"))
            await _wait_for(
                lambda: any("Approval required" in m.content for m in channel.sent)
            )
            await bus.publish_inbound(InboundMessage("fake", "7", "42", "no"))
            await _wait_for(lambda: len(channel.sent) >= 2)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return [m.content for m in channel.sent]

    contents = asyncio.run(scenario())
    assert "didn't run" in contents[-1]
    assert executed == []


def test_approval_timeout_fails_closed():
    registry = ToolRegistry()
    registry.add(
        Tool(
            lambda order_id: "expedited",
            "expedite_order",
            "Expedite an order.",
            needs_approval=True,
        )
    )

    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=0.05)
        handler = ChatHandler(
            ApprovalProvider(), registry, approvals=broker, max_steps=2
        )
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel], bus, handler.handle, approvals=broker
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(InboundMessage("fake", "7", "42", "go"))
            await _wait_for(
                lambda: any("didn't run" in m.content for m in channel.sent)
            )
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return [m.content for m in channel.sent]

    contents = asyncio.run(scenario())
    assert any("Approval required" in content for content in contents)


def test_cancel_releases_a_pending_approval():
    executed: list[str] = []

    def expedite(order_id: str) -> str:
        executed.append(order_id)
        return "expedited"

    registry = ToolRegistry()
    registry.add(
        Tool(expedite, "expedite_order", "Expedite an order.", needs_approval=True)
    )

    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        handler = ChatHandler(
            ApprovalProvider(), registry, approvals=broker, max_steps=2
        )
        channel = FakeChannel(SimpleNamespace(allow_from=("*",)), bus)
        manager = ChannelManager(
            [channel],
            bus,
            handler.handle,
            approvals=broker,
            commands=handler.handle_command,
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(lambda: channel.is_running)
            await bus.publish_inbound(_inbound("go"))
            await _wait_for(
                lambda: any("Approval required" in m.content for m in channel.sent)
            )
            await bus.publish_inbound(_inbound("/cancel"))
            # Without the broker wake-up the turn stays blocked for the full
            # five-second approval timeout, and the third message never lands.
            await _wait_for(lambda: len(channel.sent) >= 3, timeout=1.0)
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return [message.content for message in channel.sent]

    contents = asyncio.run(scenario())
    assert any("Cancelled" in content for content in contents)
    assert executed == []


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def test_cli_lists_channels(capsys):
    assert cli_main(["--list"]) == 0
    assert "telegram" in capsys.readouterr().out


def test_cli_requires_a_token(monkeypatch, capsys):
    monkeypatch.delenv("SORA_TELEGRAM_TOKEN", raising=False)
    assert cli_main(["--channel", "telegram"]) == 2
    assert "SORA_TELEGRAM_TOKEN" in capsys.readouterr().err


def test_cli_rejects_unwired_channels(capsys):
    assert cli_main(["--channel", "discord"]) == 2
    assert "no configuration" in capsys.readouterr().err
