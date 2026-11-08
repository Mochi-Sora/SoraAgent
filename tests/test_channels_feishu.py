"""Tests for the Feishu / Lark channel."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.feishu.config import FeishuConfig, from_settings
from sora_agent.channels.feishu.runtime import FeishuChannel, parse_event
from sora_agent.channels.feishu.transport import FeishuError, UrllibFeishuTransport
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> FeishuConfig:
    values = {
        "app_id": "app",
        "app_secret": "secret",
        "verification_token": "verify",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return FeishuConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.reactions: list[tuple] = []
        self.deleted: list[tuple] = []
        self.downloads: list[tuple] = []

    def send_text(self, chat_id, text):
        self.sent.append((chat_id, text))
        return "om_out"

    def add_reaction(self, message_id, emoji):
        self.reactions.append((message_id, emoji))
        return "reaction-1"

    def delete_reaction(self, message_id, reaction_id):
        self.deleted.append((message_id, reaction_id))

    def download_resource(self, message_id, file_key, resource_type, max_bytes):
        self.downloads.append((message_id, file_key, resource_type, max_bytes))
        return b"image"


def _payload(message_type="text", content=None, sender_type="user", token="verify"):
    if content is None:
        content = json.dumps({"text": "@_user_1 hello"})
    return {
        "header": {
            "event_type": "im.message.receive_v1",
            "token": token,
        },
        "event": {
            "sender": {
                "sender_type": sender_type,
                "sender_id": {"open_id": "ou_1"},
            },
            "message": {
                "message_id": "om_1",
                "chat_id": "oc_1",
                "chat_type": "p2p",
                "message_type": message_type,
                "content": content,
            },
        },
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_app_credentials_and_token():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_FEISHU_APP_ID"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_rejects_encrypted_events(monkeypatch):
    monkeypatch.setenv("SORA_FEISHU_APP_ID", "app")
    monkeypatch.setenv("SORA_FEISHU_APP_SECRET", "secret")
    monkeypatch.setenv("SORA_FEISHU_VERIFICATION_TOKEN", "verify")
    monkeypatch.setenv("SORA_FEISHU_ENCRYPT_KEY", "key")
    with pytest.raises(ValueError, match="encrypt_key"):
        from_settings(
            SimpleNamespace(
                channel=SimpleNamespace(options={}, allow_from=(), typing=True)
            ),
            {},
            SimpleNamespace(allow=None),
        )


def test_config_uses_lark_domain():
    config = _config(domain="lark")
    assert config.resolved_api_base == "https://open.larksuite.com"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_event_maps_text_and_strips_mentions():
    message = parse_event(_payload())
    assert message is not None
    assert message.content == "hello"
    assert message.session_key == "feishu:oc_1"
    assert message.metadata["message_id"] == "om_1"


def test_parse_event_maps_post_content():
    message = parse_event(
        _payload(
            message_type="post",
            content=json.dumps(
                {
                    "title": "Title",
                    "content": [
                        [{"tag": "text", "text": "hello"}],
                        [{"tag": "a", "text": "link", "href": "https://x"}],
                    ],
                }
            ),
        )
    )
    assert message is not None
    assert message.content == "Title\nhello\nlink(https://x)"


def test_parse_event_maps_image_and_file_attachments():
    image = parse_event(
        _payload(message_type="image", content=json.dumps({"image_key": "img_1"}))
    )
    assert image is not None
    assert image.attachments[0].file_id == "img_1"
    assert image.attachments[0].metadata["resource_type"] == "image"

    document = parse_event(
        _payload(
            message_type="file",
            content=json.dumps({"file_key": "file_1", "file_name": "report.pdf"}),
        )
    )
    assert document is not None
    assert document.attachments[0].filename == "report.pdf"


def test_parse_event_ignores_bot_senders_and_empty_messages():
    assert parse_event(_payload(sender_type="app")) is None
    assert parse_event(_payload(content=json.dumps({"text": ""}))) is None


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_webhook_challenge_and_token_validation():
    channel = FeishuChannel(_config(), MessageBus(), transport=FakeTransport())
    status, _headers, body = channel.handle_webhook(
        "POST",
        "/feishu",
        {},
        json.dumps(
            {"type": "url_verification", "token": "verify", "challenge": "abc"}
        ).encode(),
    )
    assert status == 200
    assert json.loads(body) == {"challenge": "abc"}
    status, _headers, _body = channel.handle_webhook(
        "POST",
        "/feishu",
        {},
        json.dumps({"type": "url_verification", "token": "bad"}).encode(),
    )
    assert status == 403


def test_webhook_rejects_bad_event_tokens():
    channel = FeishuChannel(_config(), MessageBus(), transport=FakeTransport())
    status, _headers, _body = channel.handle_webhook(
        "POST",
        "/feishu",
        {},
        json.dumps(_payload(token="bad")).encode(),
    )
    assert status == 403


def test_typing_reaction_is_added_once_and_removed():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = FeishuChannel(_config(), bus, transport=transport)
        inbound = InboundMessage(
            "feishu", "ou_1", "oc_1", "hi", metadata={"message_id": "om_1"}
        )
        await channel.send_typing(inbound)
        await channel.send_typing(inbound)
        await channel.send(
            OutboundMessage("feishu", "oc_1", "done", metadata={"message_id": "om_1"})
        )
        return transport

    transport = asyncio.run(scenario())
    assert transport.reactions == [("om_1", "OnIt")]
    assert transport.deleted == [("om_1", "reaction-1")]


def test_ingest_downloads_resources_and_dedupes():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = FeishuChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        message = parse_event(
            _payload(message_type="image", content=json.dumps({"image_key": "img_1"}))
        )
        assert message is not None
        await asyncio.to_thread(channel._ingest, message)
        await asyncio.to_thread(channel._ingest, message)
        return await asyncio.wait_for(bus.next_inbound(), 2), transport

    message, transport = asyncio.run(scenario())
    assert message.attachments[0].data == b"image"
    assert transport.downloads == [("om_1", "img_1", "image", 20 * 1024 * 1024)]


def test_send_splits_long_messages():
    async def scenario():
        transport = FakeTransport()
        channel = FeishuChannel(_config(max_message_chars=5), MessageBus(), transport=transport)
        await channel.send(OutboundMessage("feishu", "oc_1", "alpha beta"))
        return transport.sent

    assert asyncio.run(scenario()) == [
        ("oc_1", "alpha"),
        ("oc_1", "beta"),
    ]


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def test_transport_caches_token_and_sends_text(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        if url.endswith("tenant_access_token/internal"):
            return {"tenant_access_token": "token", "expire": 7200}
        return {"code": 0, "data": {"message_id": "om_out"}}

    monkeypatch.setattr(
        "sora_agent.channels.feishu.transport.request_json", fake_request
    )
    transport = UrllibFeishuTransport("app", "secret", "https://open.feishu.cn")
    assert transport.send_text("oc_1", "hi") == "om_out"
    assert transport.send_text("oc_1", "again") == "om_out"
    assert len([url for url in calls if url.endswith("tenant_access_token/internal")]) == 1
    assert calls[-1].endswith("/open-apis/im/v1/messages?receive_id_type=chat_id")


def test_transport_reports_api_errors(monkeypatch):
    def fake_request(url, **kwargs):
        if url.endswith("tenant_access_token/internal"):
            return {"tenant_access_token": "token", "expire": 7200}
        return {"code": 99991663, "msg": "no permission"}

    monkeypatch.setattr(
        "sora_agent.channels.feishu.transport.request_json", fake_request
    )
    transport = UrllibFeishuTransport("app", "secret", "https://open.feishu.cn")
    with pytest.raises(FeishuError, match="no permission"):
        transport.send_text("oc_1", "hi")
