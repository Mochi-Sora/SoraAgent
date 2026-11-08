"""Tests for the Discord channel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.discord.config import DiscordConfig, from_settings
from sora_agent.channels.discord.runtime import DiscordChannel, parse_message
from sora_agent.channels.discord.transport import (
    DiscordError,
    UrllibDiscordTransport,
)
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> DiscordConfig:
    values = {"token": "token", "allow_from": ("*",)}
    values.update(overrides)
    return DiscordConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.typing: list[str] = []
        self.downloads: list[tuple] = []

    def gateway_url(self) -> str:
        return "wss://gateway.example"

    def send_message(self, channel_id, text, *, reply_to=None):
        self.sent.append((channel_id, text, reply_to))
        return "9"

    def send_typing(self, channel_id):
        self.typing.append(channel_id)

    def download(self, url, max_bytes):
        self.downloads.append((url, max_bytes))
        return b"image"


class FakeWS:
    def __init__(self, payloads) -> None:
        self.payloads = list(payloads)
        self.sent: list[dict] = []

    def recv_json(self, timeout=None):
        if not self.payloads:
            from sora_agent.channels._ws import WebSocketClosed

            raise WebSocketClosed(1000)
        return self.payloads.pop(0)

    def send_json(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


def _message(**extra):
    data = {
        "id": "m1",
        "channel_id": "c1",
        "content": "hello",
        "author": {"id": "u1", "username": "User", "bot": False},
    }
    data.update(extra)
    return data


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_a_token():
    with pytest.raises(ValueError, match="SORA_DISCORD_TOKEN"):
        from_settings(
            SimpleNamespace(channel=SimpleNamespace(options={}, allow_from=(), typing=True)),
            {},
            SimpleNamespace(allow=None),
        )


def test_config_reads_environment_and_options(monkeypatch):
    monkeypatch.setenv("SORA_DISCORD_TOKEN", "env-token")
    monkeypatch.setenv("SORA_DISCORD_ALLOW_FROM", "1, 2")
    monkeypatch.setenv("SORA_DISCORD_MAX_CHARS", "100")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(
                options={"api_base": "https://example.test"},
                allow_from=("9",),
                typing=False,
            )
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.token == "env-token"
    assert config.allow_from == ("1", "2")
    assert config.max_message_chars == 100
    assert config.api_base == "https://example.test"
    assert config.typing is False


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="intents"):
        _config(intents=0)
    with pytest.raises(ValueError, match="max_message_chars"):
        _config(max_message_chars=0)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_message_strips_the_bot_mention():
    message = parse_message(_message(content="<@bot> hello"), "bot")
    assert message is not None
    assert message.content == "hello"
    assert message.session_key == "discord:c1"
    assert message.metadata["message_id"] == "m1"


def test_parse_message_ignores_bots_and_empty_messages():
    assert parse_message(_message(author={"id": "u", "bot": True})) is None
    assert parse_message(_message(content="")) is None


def test_parse_message_builds_attachments():
    message = parse_message(
        _message(
            content="",
            attachments=[
                {
                    "url": "https://cdn.example/x.png",
                    "filename": "x.png",
                    "content_type": "image/png",
                    "size": 4,
                }
            ],
        )
    )
    assert message is not None
    assert message.content == "The user sent attachments."
    assert message.media == ["https://cdn.example/x.png"]
    assert message.attachments[0].kind == "image"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_send_splits_long_messages():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = DiscordChannel(_config(max_message_chars=5), bus, transport=transport)
        await channel.send(OutboundMessage("discord", "c1", "alpha beta gamma"))
        return transport.sent

    sent = asyncio.run(scenario())
    assert sent == [("c1", "alpha", None), ("c1", "beta", None), ("c1", "gamma", None)]


def test_send_typing_is_best_effort():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = DiscordChannel(_config(), bus, transport=transport)
        await channel.send_typing(InboundMessage("discord", "u1", "c1", "hi"))
        return transport.typing

    assert asyncio.run(scenario()) == ["c1"]


def test_dispatch_publishes_allowed_messages_once():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = DiscordChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        await asyncio.to_thread(
            channel._dispatch, "MESSAGE_CREATE", _message()
        )
        await asyncio.to_thread(
            channel._dispatch, "MESSAGE_CREATE", _message()
        )
        message = await asyncio.wait_for(bus.next_inbound(), 2)
        return message

    message = asyncio.run(scenario())
    assert message.content == "hello"
    assert message.sender_id == "u1"


def test_dispatch_drops_unlisted_senders():
    async def scenario():
        bus = MessageBus()
        channel = DiscordChannel(
            _config(allow_from=("other",)), bus, transport=FakeTransport()
        )
        channel._loop = asyncio.get_running_loop()
        await asyncio.to_thread(
            channel._dispatch, "MESSAGE_CREATE", _message()
        )
        await asyncio.sleep(0.02)
        return bus.inbound.qsize()

    assert asyncio.run(scenario()) == 0


def test_run_socket_identifies_and_publishes():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = DiscordChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        channel._running = True
        ws = FakeWS(
            [
                {"op": 10, "d": {"heartbeat_interval": 50000}},
                {
                    "op": 0,
                    "s": 1,
                    "t": "READY",
                    "d": {"user": {"id": "bot"}},
                },
                {
                    "op": 0,
                    "s": 2,
                    "t": "MESSAGE_CREATE",
                    "d": _message(content="<@bot> hello", author={"id": "u1", "bot": False}),
                },
                {"op": 7},
            ]
        )
        task = asyncio.create_task(asyncio.to_thread(channel._run_socket, ws))
        message = await asyncio.wait_for(bus.next_inbound(), 2)
        await asyncio.wait_for(task, 2)
        return message, ws

    message, ws = asyncio.run(scenario())
    assert message.content == "hello"
    identify = next(item for item in ws.sent if item.get("op") == 2)
    assert identify["d"]["token"] == "token"


# ---------------------------------------------------------------------------
# transport payloads
# ---------------------------------------------------------------------------


def test_transport_sends_a_message(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"id": "9"}

    monkeypatch.setattr(
        "sora_agent.channels.discord.transport.request_json", fake_request
    )
    transport = UrllibDiscordTransport("token")
    assert transport.send_message("c1", "hi", reply_to="m1") == "9"
    url, kwargs = calls[0]
    assert url.endswith("/channels/c1/messages")
    assert kwargs["json_body"]["content"] == "hi"
    assert kwargs["json_body"]["message_reference"]["message_id"] == "m1"
    assert kwargs["headers"]["Authorization"] == "Bot token"


def test_transport_typing_and_gateway_url(monkeypatch):
    replies = iter([{"url": "wss://gateway"}, None])

    def fake_request(url, **kwargs):
        return next(replies)

    monkeypatch.setattr(
        "sora_agent.channels.discord.transport.request_json", fake_request
    )
    transport = UrllibDiscordTransport("token")
    assert transport.gateway_url() == "wss://gateway"
    transport.send_typing("c1")


def test_transport_wraps_http_errors(monkeypatch):
    from sora_agent.channels._http import HttpError

    def fake_request(url, **kwargs):
        raise HttpError("nope", status=401)

    monkeypatch.setattr(
        "sora_agent.channels.discord.transport.request_json", fake_request
    )
    transport = UrllibDiscordTransport("token")
    with pytest.raises(DiscordError, match="401"):
        transport.send_message("c1", "hi")


def test_transport_retries_rate_limits(monkeypatch):
    from sora_agent.channels._http import HttpError

    replies = iter([HttpError("slow down", status=429, retry_after=0), {"id": "9"}])

    def fake_request(url, **kwargs):
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        "sora_agent.channels.discord.transport.request_json", fake_request
    )
    monkeypatch.setattr("sora_agent.channels.discord.transport.time.sleep", lambda _: None)
    transport = UrllibDiscordTransport("token")
    assert transport.send_message("c1", "hi") == "9"
