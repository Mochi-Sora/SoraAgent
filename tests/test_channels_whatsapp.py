"""Tests for the WhatsApp Cloud API channel."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.types import InboundMessage, OutboundMessage
from sora_agent.channels.whatsapp.config import WhatsAppConfig, from_settings
from sora_agent.channels.whatsapp.runtime import (
    WhatsAppChannel,
    parse_message,
    parse_webhook,
)
from sora_agent.channels.whatsapp.transport import (
    UrllibWhatsAppTransport,
    WhatsAppError,
)


def _config(**overrides) -> WhatsAppConfig:
    values = {
        "phone_number_id": "123",
        "access_token": "token",
        "verify_token": "verify",
        "app_secret": "secret",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return WhatsAppConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.read: list[tuple] = []
        self.media: list[tuple] = []
        self.signature_ok = True

    def send_text(self, phone_number_id, to, text, *, reply_to=None):
        self.sent.append((phone_number_id, to, text, reply_to))
        return "wamid.out"

    def mark_read(self, phone_number_id, message_id, *, typing=True):
        self.read.append((phone_number_id, message_id, typing))

    def download_media(self, media_id, max_bytes):
        self.media.append((media_id, max_bytes))
        return b"image", "image/png"

    def verify_signature(self, body, signature):
        return self.signature_ok


def _payload(text: str | None = "hello", **message_extra):
    message = {
        "from": "15550001",
        "id": "wamid.1",
        "timestamp": "1",
        "type": "text",
        "text": {"body": text},
    }
    message.update(message_extra)
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "123"},
                            "messages": [message],
                        }
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_credentials_and_signature_secret():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_WHATSAPP_PHONE_NUMBER_ID"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_allows_explicitly_unverified_webhooks(monkeypatch):
    monkeypatch.setenv("SORA_WHATSAPP_PHONE_NUMBER_ID", "123")
    monkeypatch.setenv("SORA_WHATSAPP_ACCESS_TOKEN", "token")
    monkeypatch.setenv("SORA_WHATSAPP_VERIFY_TOKEN", "verify")
    monkeypatch.setenv("SORA_WHATSAPP_ALLOW_UNVERIFIED", "1")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.require_signature is False


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_webhook_maps_text_and_media():
    messages = parse_webhook(_payload("hello"))
    assert len(messages) == 1
    assert messages[0].content == "hello"
    assert messages[0].sender_id == "15550001"

    media = parse_webhook(
        _payload(
            type="image",
            text=None,
            image={"id": "media-1", "mime_type": "image/png", "caption": "look"},
        )
    )[0]
    assert media.content == "look"
    assert media.attachments[0].file_id == "media-1"
    assert media.attachments[0].kind == "image"


def test_parse_webhook_ignores_statuses_and_unknown_objects():
    assert parse_webhook({"object": "page", "entry": []}) == []
    payload = _payload()
    payload["entry"][0]["changes"][0]["value"].pop("messages")
    assert parse_webhook(payload) == []


def test_parse_message_handles_buttons_and_locations():
    button = parse_message(
        {
            "from": "1",
            "id": "x",
            "type": "button",
            "button": {"text": "Yes"},
        }
    )
    assert button is not None
    assert button.content == "Yes"
    location = parse_message(
        {
            "from": "1",
            "id": "x",
            "type": "location",
            "location": {"latitude": 1.5, "longitude": 2.5},
        }
    )
    assert location is not None
    assert "1.5" in location.content


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_webhook_verification_and_signature():
    transport = FakeTransport()
    channel = WhatsAppChannel(_config(), MessageBus(), transport=transport)
    status, _headers, body = channel.handle_webhook(
        "GET",
        "/whatsapp?hub.mode=subscribe&hub.verify_token=verify&hub.challenge=abc",
        {},
        b"",
    )
    assert (status, body) == (200, "abc")
    status, _headers, _body = channel.handle_webhook(
        "GET",
        "/whatsapp?hub.mode=subscribe&hub.verify_token=bad&hub.challenge=abc",
        {},
        b"",
    )
    assert status == 403
    transport.signature_ok = False
    status, _headers, _body = channel.handle_webhook(
        "POST", "/whatsapp", {"x-hub-signature-256": "bad"}, b"{}"
    )
    assert status == 403


def test_webhook_publishes_a_valid_text_message():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = WhatsAppChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        status, _headers, _body = channel.handle_webhook(
            "POST",
            "/whatsapp",
            {"x-hub-signature-256": "sig"},
            json.dumps(_payload("hello")).encode(),
        )
        message = await asyncio.wait_for(bus.next_inbound(), 2)
        return status, message

    status, message = asyncio.run(scenario())
    assert status == 200
    assert message.content == "hello"


def test_ingest_downloads_media_and_applies_the_allow_list():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = WhatsAppChannel(_config(allow_from=("15550001",)), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        message = parse_webhook(
            _payload(
                type="image",
                text=None,
                image={"id": "media-1", "mime_type": "image/png"},
            )
        )[0]
        await asyncio.to_thread(channel._ingest, message)
        published = await asyncio.wait_for(bus.next_inbound(), 2)
        return published, transport

    message, transport = asyncio.run(scenario())
    assert message.attachments[0].data == b"image"
    assert transport.media == [("media-1", 20 * 1024 * 1024)]


def test_send_and_typing_payloads():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = WhatsAppChannel(_config(max_message_chars=5), bus, transport=transport)
        await channel.send(OutboundMessage("whatsapp", "1555", "alpha beta"))
        await channel.send_typing(
            InboundMessage(
                "whatsapp", "1555", "1555", "hi", metadata={"message_id": "wamid.1"}
            )
        )
        return transport

    transport = asyncio.run(scenario())
    assert transport.sent == [
        ("123", "1555", "alpha", None),
        ("123", "1555", "beta", None),
    ]
    assert transport.read == [("123", "wamid.1", True)]


# ---------------------------------------------------------------------------
# transport payloads
# ---------------------------------------------------------------------------


def test_transport_send_and_read(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"messages": [{"id": "wamid.out"}]}

    monkeypatch.setattr(
        "sora_agent.channels.whatsapp.transport.request_json", fake_request
    )
    transport = UrllibWhatsAppTransport("token")
    assert transport.send_text("123", "1555", "hi", reply_to="wamid.1") == "wamid.out"
    transport.mark_read("123", "wamid.1", typing=True)
    assert calls[0][1]["json_body"]["text"]["body"] == "hi"
    assert calls[0][1]["json_body"]["context"]["message_id"] == "wamid.1"
    assert calls[1][1]["json_body"]["typing_indicator"] == {"type": "text"}


def test_transport_signature_validation():
    body = b'{"hello":true}'
    secret = "secret"
    signature = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    transport = UrllibWhatsAppTransport("token", app_secret=secret)
    assert transport.verify_signature(body, signature) is True
    assert transport.verify_signature(body, "sha256=bad") is False


def test_transport_wraps_errors(monkeypatch):
    from sora_agent.channels._http import HttpError

    def fake_request(url, **kwargs):
        raise HttpError("nope", status=400)

    monkeypatch.setattr(
        "sora_agent.channels.whatsapp.transport.request_json", fake_request
    )
    transport = UrllibWhatsAppTransport("token")
    with pytest.raises(WhatsAppError):
        transport.send_text("123", "1555", "hi")
