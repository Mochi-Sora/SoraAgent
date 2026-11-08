"""Tests for the Signal channel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.signal.config import SignalConfig, from_settings
from sora_agent.channels.signal.runtime import SignalChannel, parse_event
from sora_agent.channels.signal.transport import (
    SignalError,
    UrllibSignalTransport,
    parse_event_lines,
)
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> SignalConfig:
    values = {
        "http_url": "http://127.0.0.1:8080",
        "account": "+15550001",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return SignalConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.typing: list[tuple] = []
        self.closed = False

    def events(self):
        return iter(())

    def send_text(self, account, recipient, text):
        self.sent.append((account, recipient, text))

    def send_typing(self, account, recipient):
        self.typing.append((account, recipient))

    def close(self):
        self.closed = True


def _event(**data_extra):
    data = {"message": "hello", "timestamp": 1}
    data.update(data_extra)
    return {
        "params": {
            "envelope": {
                "sourceNumber": "+15550002",
                "timestamp": 1,
                "dataMessage": data,
            }
        }
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_url_and_account():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_SIGNAL_HTTP_URL"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("SORA_SIGNAL_HTTP_URL", "http://signal.example")
    monkeypatch.setenv("SORA_SIGNAL_ACCOUNT", "+1555")
    monkeypatch.setenv("SORA_SIGNAL_ALLOW_FROM", "+1, +2")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.http_url == "http://signal.example"
    assert config.account == "+1555"
    assert config.allow_from == ("+1", "+2")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_event_maps_direct_and_group_messages():
    direct = parse_event(_event())
    assert direct is not None
    assert direct.content == "hello"
    assert direct.session_key == "signal:+15550002"

    group = parse_event(
        _event(groupInfo={"groupId": "group-1", "groupName": "Team"})
    )
    assert group is not None
    assert group.chat_id == "group-1"
    assert group.sender_id == "+15550002"


def test_parse_event_ignores_receipts_and_empty_messages():
    assert parse_event({"params": {"envelope": {"receiptMessage": {}}}}) is None
    assert parse_event(_event(message="")) is None


def test_parse_event_keeps_attachment_metadata():
    message = parse_event(
        _event(
            message="",
            attachments=[
                {
                    "id": "att-1",
                    "filename": "photo.jpg",
                    "contentType": "image/jpeg",
                    "size": 10,
                }
            ],
        )
    )
    assert message is not None
    assert message.content == "The user sent attachments."
    assert message.attachments[0].file_id == "att-1"
    assert message.attachments[0].kind == "image"


def test_parse_event_lines_reads_sse():
    events = parse_event_lines(
        [
            'data: {"params": {"envelope": {"dataMessage": {"message": "hi"}}}}',
            "",
        ]
    )
    assert events[0]["params"]["envelope"]["dataMessage"]["message"] == "hi"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_dispatch_publishes_allowed_messages_once():
    async def scenario():
        bus = MessageBus()
        channel = SignalChannel(_config(), bus, transport=FakeTransport())
        channel._loop = asyncio.get_running_loop()
        await asyncio.to_thread(channel._dispatch, _event())
        await asyncio.to_thread(channel._dispatch, _event())
        return await asyncio.wait_for(bus.next_inbound(), 2)

    message = asyncio.run(scenario())
    assert message.content == "hello"


def test_dispatch_drops_unlisted_senders():
    async def scenario():
        bus = MessageBus()
        channel = SignalChannel(
            _config(allow_from=("+1",)), bus, transport=FakeTransport()
        )
        channel._loop = asyncio.get_running_loop()
        await asyncio.to_thread(channel._dispatch, _event())
        await asyncio.sleep(0.02)
        return bus.inbound.qsize()

    assert asyncio.run(scenario()) == 0


def test_send_and_typing():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = SignalChannel(_config(max_message_chars=5), bus, transport=transport)
        await channel.send(OutboundMessage("signal", "+1555", "alpha beta"))
        await channel.send_typing(InboundMessage("signal", "+1555", "+1555", "hi"))
        return transport

    transport = asyncio.run(scenario())
    assert transport.sent == [
        ("+15550001", "+1555", "alpha"),
        ("+15550001", "+1555", "beta"),
    ]
    assert transport.typing == [("+15550001", "+1555")]


def test_stop_closes_the_transport():
    async def scenario():
        transport = FakeTransport()
        channel = SignalChannel(_config(), MessageBus(), transport=transport)
        await channel.stop()
        return transport.closed

    assert asyncio.run(scenario()) is True


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def test_transport_sends_and_types(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"result": {"timestamp": 1}}

    monkeypatch.setattr(
        "sora_agent.channels.signal.transport.request_json", fake_request
    )
    transport = UrllibSignalTransport("http://signal", "+1555")
    transport.send_text("+1555", "+1556", "hi")
    transport.send_typing("+1555", "+1556")
    assert calls[0][1]["json_body"]["method"] == "send"
    assert calls[0][1]["json_body"]["params"]["recipient"] == ["+1556"]
    assert calls[1][1]["json_body"]["method"] == "sendTyping"


def test_transport_raises_on_rpc_errors(monkeypatch):
    def fake_request(url, **kwargs):
        return {"error": {"message": "typing unsupported"}}

    monkeypatch.setattr(
        "sora_agent.channels.signal.transport.request_json", fake_request
    )
    transport = UrllibSignalTransport("http://signal", "+1555")
    with pytest.raises(SignalError, match="typing unsupported"):
        transport.send_typing("+1555", "+1556")
