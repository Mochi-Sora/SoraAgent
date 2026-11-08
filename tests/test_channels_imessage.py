"""Tests for the iMessage (BlueBubbles) channel."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.imessage.config import IMessageConfig, from_settings
from sora_agent.channels.imessage.runtime import IMessageChannel, parse_event
from sora_agent.channels.imessage.transport import (
    IMessageError,
    UrllibIMessageTransport,
)
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> IMessageConfig:
    values = {
        "server_url": "http://127.0.0.1:1234",
        "password": "secret",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return IMessageConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.typing: list[str] = []

    def send_text(self, chat_guid, text):
        self.sent.append((chat_guid, text))
        return "guid"

    def send_typing(self, chat_guid):
        self.typing.append(chat_guid)


def _payload(**data_extra):
    data = {
        "guid": "m1",
        "text": "hello",
        "isFromMe": False,
        "handle": {"address": "+15550001"},
        "chats": [{"guid": "iMessage;-;+15550001"}],
        "attachments": [],
    }
    data.update(data_extra)
    return {"type": "new-message", "data": data}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_server_and_password():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_IMESSAGE_SERVER_URL"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("SORA_IMESSAGE_SERVER_URL", "http://blue.example")
    monkeypatch.setenv("SORA_IMESSAGE_PASSWORD", "env-secret")
    monkeypatch.setenv("SORA_IMESSAGE_ALLOW_FROM", "+1, +2")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.server_url == "http://blue.example"
    assert config.password == "env-secret"
    assert config.allow_from == ("+1", "+2")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_event_maps_text_and_chat_guid():
    message = parse_event(_payload())
    assert message is not None
    assert message.content == "hello"
    assert message.session_key == "imessage:iMessage;-;+15550001"
    assert message.sender_id == "+15550001"


def test_parse_event_ignores_self_reactions_and_empty_messages():
    assert parse_event(_payload(isFromMe=True)) is None
    assert parse_event(_payload(associatedMessageGuid="parent")) is None
    assert parse_event(_payload(text="")) is None


def test_parse_event_maps_attachments():
    message = parse_event(
        _payload(
            text="",
            attachments=[
                {
                    "transferName": "photo.jpg",
                    "mimeType": "image/jpeg",
                    "totalBytes": 10,
                    "guid": "att-1",
                }
            ],
        )
    )
    assert message is not None
    assert message.content == "The user sent attachments."
    assert message.attachments[0].kind == "image"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_webhook_secret_is_enforced():
    channel = IMessageChannel(
        _config(webhook_secret="hook-secret"),
        MessageBus(),
        transport=FakeTransport(),
    )
    status, _headers, _body = channel.handle_webhook(
        "POST", "/imessage", {}, json.dumps(_payload()).encode()
    )
    assert status == 403
    status, _headers, _body = channel.handle_webhook(
        "POST",
        "/imessage",
        {"x-webhook-secret": "hook-secret"},
        json.dumps(_payload()).encode(),
    )
    assert status == 200


def test_webhook_rejects_invalid_json():
    channel = IMessageChannel(
        _config(), MessageBus(), transport=FakeTransport()
    )
    status, _headers, _body = channel.handle_webhook(
        "POST", "/imessage", {}, b"not-json"
    )
    assert status == 400


def test_ingest_publishes_allowed_messages_once():
    async def scenario():
        bus = MessageBus()
        channel = IMessageChannel(_config(), bus, transport=FakeTransport())
        channel._loop = asyncio.get_running_loop()
        message = parse_event(_payload())
        assert message is not None
        await asyncio.to_thread(channel._ingest, message)
        await asyncio.to_thread(channel._ingest, message)
        return await asyncio.wait_for(bus.next_inbound(), 2)

    message = asyncio.run(scenario())
    assert message.content == "hello"


def test_send_and_typing():
    async def scenario():
        transport = FakeTransport()
        channel = IMessageChannel(
            _config(max_message_chars=5), MessageBus(), transport=transport
        )
        await channel.send(OutboundMessage("imessage", "iMessage;-;+1555", "alpha beta"))
        await channel.send_typing(
            InboundMessage("imessage", "+1555", "iMessage;-;+1555", "hi")
        )
        return transport

    transport = asyncio.run(scenario())
    assert transport.sent == [
        ("iMessage;-;+1555", "alpha"),
        ("iMessage;-;+1555", "beta"),
    ]
    assert transport.typing == ["iMessage;-;+1555"]


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def test_transport_sends_text_and_typing(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"status": 200, "data": {"guid": "message-guid"}}

    monkeypatch.setattr(
        "sora_agent.channels.imessage.transport.request_json", fake_request
    )
    transport = UrllibIMessageTransport("http://blue.example", "secret")
    assert transport.send_text("iMessage;-;+1555", "hi") == "message-guid"
    transport.send_typing("iMessage;-;+1555")
    assert "password=secret" in calls[0][0]
    assert calls[0][1]["json_body"]["chatGuid"] == "iMessage;-;+1555"
    assert "typing" in calls[1][0]
    assert calls[1][1]["json_body"] == {"typing": True}


def test_transport_reports_api_errors(monkeypatch):
    def fake_request(url, **kwargs):
        return {"status": 500, "message": "boom"}

    monkeypatch.setattr(
        "sora_agent.channels.imessage.transport.request_json", fake_request
    )
    transport = UrllibIMessageTransport("http://blue.example", "secret")
    with pytest.raises(IMessageError, match="boom"):
        transport.send_text("chat", "hi")
