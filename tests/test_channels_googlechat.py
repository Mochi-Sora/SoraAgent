"""Tests for the Google Chat channel."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.googlechat.config import GoogleChatConfig, from_settings
from sora_agent.channels.googlechat.runtime import GoogleChatChannel, parse_event
from sora_agent.channels.googlechat.transport import (
    GoogleChatError,
    UrllibGoogleChatTransport,
)
from sora_agent.channels.types import OutboundMessage


def _config(**overrides) -> GoogleChatConfig:
    values = {
        "webhook_url": "https://chat.example/webhook",
        "audience": "aud",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return GoogleChatConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.verify_ok = True

    def send_webhook(self, text, *, thread_name=None):
        self.sent.append((text, thread_name))

    def verify_jwt(self, token):
        if not self.verify_ok:
            raise GoogleChatError("bad token")
        return {}


def _event(**message_extra):
    message = {
        "name": "spaces/S/messages/M",
        "messageId": "M",
        "text": "@Sora hello",
        "argumentText": " hello",
        "thread": {"name": "spaces/S/threads/T"},
        "attachment": [],
    }
    message.update(message_extra)
    return {
        "type": "MESSAGE",
        "message": message,
        "sender": {"name": "users/U", "displayName": "User", "type": "HUMAN"},
        "space": {"name": "spaces/S", "type": "ROOM"},
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_webhook_and_audience():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_GOOGLE_CHAT_WEBHOOK_URL"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_allows_unverified_requests(monkeypatch):
    monkeypatch.setenv("SORA_GOOGLE_CHAT_WEBHOOK_URL", "https://chat.example/hook")
    monkeypatch.setenv("SORA_GOOGLE_CHAT_ALLOW_UNVERIFIED", "1")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.verify_jwt is False


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_event_prefers_argument_text_and_thread():
    message = parse_event(_event())
    assert message is not None
    assert message.content == "hello"
    assert message.session_key == "googlechat:spaces/S:spaces/S/threads/T"
    assert message.metadata["thread_name"] == "spaces/S/threads/T"


def test_parse_event_ignores_bots_and_empty_messages():
    payload = _event()
    payload["sender"]["type"] = "BOT"
    assert parse_event(payload) is None
    assert parse_event(_event(text="", argumentText="")) is None


def test_parse_event_maps_attachments():
    message = parse_event(
        _event(
            text="",
            argumentText="",
            attachment=[
                {
                    "name": "photo.jpg",
                    "contentType": "image/jpeg",
                    "downloadUri": "https://chat.example/photo",
                }
            ],
        )
    )
    assert message is not None
    assert message.attachments[0].file_id == "https://chat.example/photo"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_webhook_gates_on_jwt_and_acknowledges():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = GoogleChatChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        status, _headers, _body = channel.handle_webhook(
            "POST", "/googlechat", {}, json.dumps(_event()).encode()
        )
        assert status == 401
        status, _headers, body = channel.handle_webhook(
            "POST",
            "/googlechat",
            {"authorization": "Bearer good"},
            json.dumps(_event()).encode(),
        )
        message = await asyncio.wait_for(bus.next_inbound(), 2)
        return status, json.loads(body), message

    status, body, message = asyncio.run(scenario())
    assert status == 200
    assert body["text"].startswith("⏳")
    assert body["thread"]["name"] == "spaces/S/threads/T"
    assert message.content == "hello"


def test_webhook_rejects_bad_tokens():
    transport = FakeTransport()
    transport.verify_ok = False
    channel = GoogleChatChannel(
        _config(), MessageBus(), transport=transport
    )
    status, _headers, _body = channel.handle_webhook(
        "POST",
        "/googlechat",
        {"authorization": "Bearer bad"},
        json.dumps(_event()).encode(),
    )
    assert status == 401


def test_webhook_can_disable_the_acknowledgement():
    channel = GoogleChatChannel(
        _config(verify_jwt=False, typing=False),
        MessageBus(),
        transport=FakeTransport(),
    )
    status, _headers, body = channel.handle_webhook(
        "POST", "/googlechat", {}, json.dumps(_event()).encode()
    )
    assert status == 200
    assert json.loads(body) == {}


def test_send_uses_the_webhook_and_thread():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = GoogleChatChannel(_config(max_message_chars=5), bus, transport=transport)
        await channel.send(
            OutboundMessage(
                "googlechat",
                "spaces/S",
                "alpha beta",
                metadata={"thread_name": "spaces/S/threads/T"},
            )
        )
        return transport.sent

    assert asyncio.run(scenario()) == [
        ("alpha", "spaces/S/threads/T"),
        ("beta", "spaces/S/threads/T"),
    ]


def test_ingest_respects_the_allow_list():
    async def scenario():
        bus = MessageBus()
        channel = GoogleChatChannel(
            _config(allow_from=("users/other",)),
            bus,
            transport=FakeTransport(),
        )
        channel._loop = asyncio.get_running_loop()
        message = parse_event(_event())
        assert message is not None
        await asyncio.to_thread(channel._ingest, message)
        await asyncio.sleep(0.02)
        return bus.inbound.qsize()

    assert asyncio.run(scenario()) == 0


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def test_transport_posts_the_webhook_payload(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {}

    monkeypatch.setattr(
        "sora_agent.channels.googlechat.transport.request_json", fake_request
    )
    transport = UrllibGoogleChatTransport(
        "https://chat.example/hook",
        audience="aud",
        issuer="issuer",
        jwks_url="https://keys.example",
    )
    transport.send_webhook("hi", thread_name="spaces/S/threads/T")
    body = calls[0][1]["json_body"]
    assert body == {"text": "hi", "thread": {"name": "spaces/S/threads/T"}}


def test_transport_wraps_webhook_errors(monkeypatch):
    from sora_agent.channels._http import HttpError

    def fake_request(url, **kwargs):
        raise HttpError("nope", status=500)

    monkeypatch.setattr(
        "sora_agent.channels.googlechat.transport.request_json", fake_request
    )
    transport = UrllibGoogleChatTransport(
        "https://chat.example/hook",
        audience="aud",
        issuer="issuer",
        jwks_url="https://keys.example",
    )
    with pytest.raises(GoogleChatError):
        transport.send_webhook("hi")
