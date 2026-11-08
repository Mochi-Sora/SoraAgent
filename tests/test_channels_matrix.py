"""Tests for the Matrix channel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.matrix.config import MatrixConfig, from_settings
from sora_agent.channels.matrix.runtime import MatrixChannel, parse_event, parse_sync
from sora_agent.channels.matrix.transport import (
    MatrixError,
    UrllibMatrixTransport,
)
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> MatrixConfig:
    values = {
        "homeserver": "https://matrix.example",
        "access_token": "token",
        "user_id": "@bot:example",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return MatrixConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.typing: list[tuple] = []
        self.media: list[tuple] = []

    def whoami(self):
        return "@bot:example"

    def sync(self, since, timeout):
        return {"next_batch": "n", "rooms": {"join": {}}}

    def send_message(self, room_id, text, *, thread_root=None, reply_to=None, txn_id=None):
        self.sent.append((room_id, text, thread_root, reply_to))
        return "$event"

    def send_typing(self, room_id, user_id, timeout_ms=30000):
        self.typing.append((room_id, user_id))

    def download_media(self, mxc_url, max_bytes):
        self.media.append((mxc_url, max_bytes))
        return b"image"


def _event(**content_extra):
    content = {"msgtype": "m.text", "body": "hello"}
    content.update(content_extra)
    return {
        "type": "m.room.message",
        "sender": "@user:example",
        "event_id": "$event",
        "content": content,
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_homeserver_and_token():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_MATRIX_HOMESERVER"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("SORA_MATRIX_HOMESERVER", "https://matrix.example")
    monkeypatch.setenv("SORA_MATRIX_ACCESS_TOKEN", "env-token")
    monkeypatch.setenv("SORA_MATRIX_USER_ID", "@bot:example")
    monkeypatch.setenv("SORA_MATRIX_ALLOW_FROM", "@u1:example, @u2:example")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.access_token == "env-token"
    assert config.user_id == "@bot:example"
    assert config.allow_from == ("@u1:example", "@u2:example")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_event_maps_text_and_threads():
    direct = parse_event(_event(), "!room:example", "@bot:example")
    assert direct is not None
    assert direct.content == "hello"
    assert direct.session_key == "matrix:!room:example"

    threaded = parse_event(
        _event(**{"m.relates_to": {"rel_type": "m.thread", "event_id": "$root"}}),
        "!room:example",
        "@bot:example",
    )
    assert threaded is not None
    assert threaded.session_key == "matrix:!room:example:$root"
    assert threaded.metadata["thread_root"] == "$root"


def test_parse_event_ignores_own_and_edited_messages():
    own = _event()
    own["sender"] = "@bot:example"
    assert parse_event(own, "!room", "@bot:example") is None
    assert parse_event(
        _event(**{"m.relates_to": {"rel_type": "m.replace", "event_id": "$x"}}),
        "!room",
        "@bot:example",
    ) is None


def test_parse_event_keeps_media_and_flags_encrypted_attachments():
    media = parse_event(
        _event(msgtype="m.image", body="photo.jpg", url="mxc://server/id", info={"mimetype": "image/jpeg", "size": 10}),
        "!room",
        "@bot:example",
    )
    assert media is not None
    assert media.attachments[0].file_id == "mxc://server/id"
    assert media.attachments[0].kind == "image"

    encrypted = parse_event(
        _event(msgtype="m.file", body="secret.pdf", file={"url": "mxc://server/x"}),
        "!room",
        "@bot:example",
    )
    assert encrypted is not None
    assert "encrypted" in encrypted.attachments[0].error


def test_parse_sync_walks_joined_rooms():
    payload = {
        "rooms": {
            "join": {
                "!room": {
                    "timeline": {"events": [_event()]},
                }
            }
        }
    }
    messages = parse_sync(payload, "@bot:example")
    assert len(messages) == 1
    assert messages[0].chat_id == "!room"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_send_and_typing_payloads():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = MatrixChannel(_config(max_message_chars=5), bus, transport=transport)
        await channel.send(
            OutboundMessage(
                "matrix",
                "!room",
                "alpha beta",
                metadata={"thread_root": "$root"},
            )
        )
        await channel.send_typing(
            InboundMessage("matrix", "@u", "!room", "hi", metadata={"message_id": "$e"})
        )
        return transport

    transport = asyncio.run(scenario())
    assert transport.sent == [
        ("!room", "alpha", "$root", None),
        ("!room", "beta", "$root", None),
    ]
    assert transport.typing == [("!room", "@bot:example")]


def test_ingest_downloads_media_and_dedupes():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = MatrixChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        message = parse_event(
            _event(
                msgtype="m.image",
                body="photo.jpg",
                url="mxc://server/id",
                info={"mimetype": "image/jpeg", "size": 10},
            ),
            "!room",
            "@bot:example",
        )
        assert message is not None
        await asyncio.to_thread(channel._ingest, message)
        await asyncio.to_thread(channel._ingest, message)
        published = await asyncio.wait_for(bus.next_inbound(), 2)
        return published, transport

    message, transport = asyncio.run(scenario())
    assert message.attachments[0].data == b"image"
    assert transport.media == [("mxc://server/id", 20 * 1024 * 1024)]


def test_send_typing_waits_for_a_user_id():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = MatrixChannel(_config(user_id=""), bus, transport=transport)
        await channel.send_typing(InboundMessage("matrix", "@u", "!room", "hi"))
        return transport.typing

    assert asyncio.run(scenario()) == []


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def test_transport_sync_and_send(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"next_batch": "n", "event_id": "$event"}

    monkeypatch.setattr(
        "sora_agent.channels.matrix.transport.request_json", fake_request
    )
    transport = UrllibMatrixTransport("https://matrix.example", "token")
    assert transport.sync("since", 30)["next_batch"] == "n"
    assert "since=since" in calls[0][0]
    assert "timeout=30000" in calls[0][0]
    assert transport.send_message(
        "!room", "hi", thread_root="$root", reply_to="$reply"
    ) == "$event"
    body = calls[1][1]["json_body"]
    assert body["m.relates_to"]["rel_type"] == "m.thread"


def test_transport_typing_and_whoami(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"user_id": "@bot:example"}

    monkeypatch.setattr(
        "sora_agent.channels.matrix.transport.request_json", fake_request
    )
    transport = UrllibMatrixTransport("https://matrix.example", "token")
    assert transport.whoami() == "@bot:example"
    transport.send_typing("!room", "@bot:example")
    assert unquote(calls[1][0]).endswith("/typing/@bot:example")
    assert calls[1][1]["json_body"]["typing"] is True


def test_transport_wraps_errors(monkeypatch):
    from sora_agent.channels._http import HttpError

    def fake_request(url, **kwargs):
        raise HttpError("nope", status=403)

    monkeypatch.setattr(
        "sora_agent.channels.matrix.transport.request_json", fake_request
    )
    transport = UrllibMatrixTransport("https://matrix.example", "token")
    with pytest.raises(MatrixError):
        transport.whoami()
