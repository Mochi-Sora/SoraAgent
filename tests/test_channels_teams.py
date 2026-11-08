"""Tests for the Microsoft Teams channel."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.teams.config import TeamsConfig, from_settings
from sora_agent.channels.teams.runtime import TeamsChannel, parse_activity
from sora_agent.channels.teams.transport import TeamsError, UrllibTeamsTransport
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> TeamsConfig:
    values = {
        "app_id": "app",
        "app_password": "secret",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return TeamsConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.typing: list[tuple] = []
        self.downloads: list[tuple] = []

    def access_token(self) -> str:
        return "token"

    def send_text(self, service_url, conversation_id, text, *, reply_to=None):
        self.sent.append((service_url, conversation_id, text, reply_to))
        return "activity"

    def send_typing(self, service_url, conversation_id):
        self.typing.append((service_url, conversation_id))

    def verify_jwt(self, token, service_url=None):
        if token != "good":
            raise TeamsError("bad token")
        return {"serviceUrl": service_url}

    def download(self, url, max_bytes):
        self.downloads.append((url, max_bytes))
        return b"file"


def _activity(**extra):
    activity = {
        "type": "message",
        "id": "a1",
        "text": "<at>Sora</at> hello",
        "from": {"id": "u1", "name": "User"},
        "conversation": {"id": "c1"},
        "serviceUrl": "https://smba.example",
        "channelId": "msteams",
    }
    activity.update(extra)
    return activity


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_credentials():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_TEAMS_APP_ID"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("SORA_TEAMS_APP_ID", "env-app")
    monkeypatch.setenv("SORA_TEAMS_APP_PASSWORD", "env-secret")
    monkeypatch.setenv("SORA_TEAMS_ALLOW_FROM", "u1, u2")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.app_id == "env-app"
    assert config.app_password == "env-secret"
    assert config.allow_from == ("u1", "u2")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_activity_strips_mentions_and_keeps_service_url():
    message = parse_activity(_activity(), "app")
    assert message is not None
    assert message.content == "hello"
    assert message.session_key == "teams:c1"
    assert message.metadata["service_url"] == "https://smba.example"


def test_parse_activity_ignores_bots_and_html_only_attachments():
    assert parse_activity(_activity(**{"from": {"role": "bot"}}), "app") is None
    assert parse_activity(
        _activity(
            text="",
            attachments=[
                {"contentType": "text/html", "content": "<p>hello</p>"}
            ],
        ),
        "app",
    ) is None


def test_parse_activity_keeps_file_attachments():
    message = parse_activity(
        _activity(
            text="",
            attachments=[
                {
                    "contentType": "application/pdf",
                    "name": "report.pdf",
                    "contentUrl": "https://files.example/report",
                }
            ],
        ),
        "app",
    )
    assert message is not None
    assert message.content == "The user sent attachments."
    assert message.attachments[0].kind == "document"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_handle_webhook_requires_a_valid_bearer_token():
    channel = TeamsChannel(
        _config(verify_jwt=True),
        MessageBus(),
        transport=FakeTransport(),
    )
    status, _headers, _body = channel.handle_webhook(
        "POST", "/teams", {}, json.dumps(_activity()).encode()
    )
    assert status == 401
    status, _headers, _body = channel.handle_webhook(
        "POST",
        "/teams",
        {"authorization": "Bearer bad"},
        json.dumps(_activity()).encode(),
    )
    assert status == 401
    status, _headers, _body = channel.handle_webhook(
        "POST",
        "/teams",
        {"authorization": "Bearer good"},
        json.dumps(_activity()).encode(),
    )
    assert status == 200


def test_handle_webhook_rejects_invalid_json():
    channel = TeamsChannel(
        _config(verify_jwt=False), MessageBus(), transport=FakeTransport()
    )
    status, _headers, _body = channel.handle_webhook(
        "POST", "/teams", {}, b"not-json"
    )
    assert status == 400


def test_ingest_publishes_allowed_messages_once():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = TeamsChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        message = parse_activity(_activity(), "app")
        assert message is not None
        await asyncio.to_thread(channel._ingest, message)
        await asyncio.to_thread(channel._ingest, message)
        published = await asyncio.wait_for(bus.next_inbound(), 2)
        return published

    message = asyncio.run(scenario())
    assert message.content == "hello"


def test_send_and_typing_use_activity_metadata():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = TeamsChannel(_config(max_message_chars=5), bus, transport=transport)
        message = OutboundMessage(
            "teams",
            "c1",
            "alpha beta",
            metadata={"service_url": "https://smba.example", "conversation_id": "c1"},
        )
        await channel.send(message)
        inbound = InboundMessage(
            "teams",
            "u1",
            "c1",
            "hi",
            metadata={"service_url": "https://smba.example"},
        )
        await channel.send_typing(inbound)
        return transport

    transport = asyncio.run(scenario())
    assert transport.sent == [
        ("https://smba.example", "c1", "alpha", None),
        ("https://smba.example", "c1", "beta", None),
    ]
    assert transport.typing == [("https://smba.example", "c1")]


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def test_transport_caches_the_access_token(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        return {"access_token": "token", "expires_in": 3600}

    monkeypatch.setattr(
        "sora_agent.channels.teams.transport.request_json", fake_request
    )
    transport = UrllibTeamsTransport(
        "app",
        "secret",
        token_url="https://login.example/token",
        openid_config_url="https://openid.example",
        scope="scope",
    )
    assert transport.access_token() == "token"
    assert transport.access_token() == "token"
    assert len(calls) == 1


def test_transport_sends_message_and_typing(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"id": "a1"}

    monkeypatch.setattr(
        "sora_agent.channels.teams.transport.request_json", fake_request
    )
    transport = UrllibTeamsTransport(
        "app",
        "secret",
        token_url="https://login.example/token",
        openid_config_url="https://openid.example",
        scope="scope",
    )
    # Avoid a real token request.
    transport._token = "token"
    transport._token_expires_at = 9_999_999_999
    assert transport.send_text("https://smba.example", "c1", "hi", reply_to="a0") == "a1"
    transport.send_typing("https://smba.example", "c1")
    assert calls[0][0] == "https://smba.example/v3/conversations/c1/activities"
    assert calls[0][1]["json_body"]["replyToId"] == "a0"
    assert calls[1][1]["json_body"]["type"] == "typing"
