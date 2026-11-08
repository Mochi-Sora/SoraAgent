"""Tests for the Slack channel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sora_agent.channels.bus import MessageBus
from sora_agent.channels.slack.config import SlackConfig, from_settings
from sora_agent.channels.slack.runtime import SlackChannel, parse_event
from sora_agent.channels.slack.transport import SlackError, UrllibSlackTransport
from sora_agent.channels.types import InboundMessage, OutboundMessage


def _config(**overrides) -> SlackConfig:
    values = {
        "bot_token": "xoxb-token",
        "app_token": "xapp-token",
        "allow_from": ("*",),
    }
    values.update(overrides)
    return SlackConfig(**values)


class FakeTransport:
    def __init__(self) -> None:
        self.posts: list[tuple] = []
        self.deleted: list[tuple] = []
        self.statuses: list[tuple] = []
        self.fail_status = False

    def open_socket(self) -> str:
        return "wss://slack.example"

    def auth_test(self):
        return {"user_id": "BOT"}

    def post_message(self, channel, text, *, thread_ts=None):
        self.posts.append((channel, text, thread_ts))
        return f"ts{len(self.posts)}"

    def delete_message(self, channel, ts):
        self.deleted.append((channel, ts))

    def set_status(self, channel, thread_ts, status):
        if self.fail_status:
            raise SlackError("missing_scope")
        self.statuses.append((channel, thread_ts, status))

    def download(self, url, max_bytes):
        return b"file"


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


def _event(**extra):
    event = {
        "type": "message",
        "user": "U1",
        "channel": "C1",
        "text": "hello",
        "ts": "1.1",
    }
    event.update(extra)
    return event


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_both_tokens():
    settings = SimpleNamespace(
        channel=SimpleNamespace(options={}, allow_from=(), typing=True)
    )
    with pytest.raises(ValueError, match="SORA_SLACK_BOT_TOKEN"):
        from_settings(settings, {}, SimpleNamespace(allow=None))


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("SORA_SLACK_BOT_TOKEN", "xoxb-env")
    monkeypatch.setenv("SORA_SLACK_APP_TOKEN", "xapp-env")
    monkeypatch.setenv("SORA_SLACK_ALLOW_FROM", "U1, U2")
    monkeypatch.setenv("SORA_SLACK_TYPING_MODE", "status")
    config = from_settings(
        SimpleNamespace(
            channel=SimpleNamespace(options={}, allow_from=(), typing=True)
        ),
        {},
        SimpleNamespace(allow=None),
    )
    assert config.bot_token == "xoxb-env"
    assert config.app_token == "xapp-env"
    assert config.allow_from == ("U1", "U2")
    assert config.typing_mode == "status"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_event_strips_mentions_and_scopes_threads():
    message = parse_event(
        _event(text="<@BOT> hello", thread_ts="1.0"),
        "BOT",
    )
    assert message is not None
    assert message.content == "hello"
    assert message.session_key == "slack:C1:1.0"
    assert message.metadata["message_id"] == "1.1"


def test_parse_event_ignores_bots_subtypes_and_empty_messages():
    assert parse_event(_event(bot_id="B"), "BOT") is None
    assert parse_event(_event(subtype="message_changed"), "BOT") is None
    assert parse_event(_event(text=""), "BOT") is None
    assert parse_event(_event(user="BOT"), "BOT") is None


def test_parse_event_accepts_file_share():
    message = parse_event(
        _event(
            subtype="file_share",
            text="",
            files=[
                {
                    "name": "report.pdf",
                    "mimetype": "application/pdf",
                    "size": 12,
                    "url_private_download": "https://files.example/report",
                }
            ],
        ),
        "BOT",
    )
    assert message is not None
    assert message.content == "The user sent attachments."
    assert message.attachments[0].kind == "document"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


def test_send_typing_posts_one_placeholder_and_send_cleans_it_up():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = SlackChannel(_config(), bus, transport=transport)
        inbound = InboundMessage(
            "slack", "U1", "C1", "hi", metadata={"message_id": "1.1"}
        )
        await channel.send_typing(inbound)
        await channel.send_typing(inbound)
        await channel.send(OutboundMessage("slack", "C1", "done"))
        return transport

    transport = asyncio.run(scenario())
    assert len(transport.posts) == 2  # placeholder + final
    assert transport.posts[0][1].startswith("⏳")
    assert transport.deleted == [("C1", "ts1")]


def test_send_typing_uses_status_in_threads():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = SlackChannel(_config(typing_mode="status"), bus, transport=transport)
        inbound = InboundMessage(
            "slack",
            "U1",
            "C1",
            "hi",
            metadata={"message_id": "1.1", "thread_ts": "1.0"},
        )
        await channel.send_typing(inbound)
        await channel.send(OutboundMessage("slack", "C1", "done", metadata={"thread_ts": "1.0"}))
        return transport

    transport = asyncio.run(scenario())
    assert transport.statuses[0] == ("C1", "1.0", "⏳ Preparing a response…")
    assert transport.statuses[-1] == ("C1", "1.0", "")
    assert transport.posts == [("C1", "done", "1.0")]


def test_send_typing_falls_back_when_status_is_unavailable():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        transport.fail_status = True
        channel = SlackChannel(_config(typing_mode="auto"), bus, transport=transport)
        inbound = InboundMessage(
            "slack",
            "U1",
            "C1",
            "hi",
            metadata={"message_id": "1.1", "thread_ts": "1.0"},
        )
        await channel.send_typing(inbound)
        return transport

    transport = asyncio.run(scenario())
    assert transport.posts[0][1].startswith("⏳")


def test_run_socket_acknowledges_and_publishes():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = SlackChannel(_config(), bus, transport=transport)
        channel._loop = asyncio.get_running_loop()
        channel._running = True
        channel._bot_user_id = "BOT"
        ws = FakeWS(
            [
                {"type": "hello"},
                {
                    "type": "events_api",
                    "envelope_id": "e1",
                    "payload": {"event": _event(text="<@BOT> hello")},
                },
                {"type": "disconnect"},
            ]
        )
        task = asyncio.create_task(asyncio.to_thread(channel._run_socket, ws))
        message = await asyncio.wait_for(bus.next_inbound(), 2)
        await asyncio.wait_for(task, 2)
        return message, ws

    message, ws = asyncio.run(scenario())
    assert message.content == "hello"
    assert ws.sent[0] == {"envelope_id": "e1"}


def test_dispatch_drops_unlisted_senders():
    async def scenario():
        bus = MessageBus()
        channel = SlackChannel(
            _config(allow_from=("OTHER",)), bus, transport=FakeTransport()
        )
        channel._loop = asyncio.get_running_loop()
        await asyncio.to_thread(channel._dispatch, _event())
        await asyncio.sleep(0.02)
        return bus.inbound.qsize()

    assert asyncio.run(scenario()) == 0


# ---------------------------------------------------------------------------
# transport payloads
# ---------------------------------------------------------------------------


def test_transport_post_and_status(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("chat.postMessage"):
            return {"ok": True, "ts": "2.0"}
        return {"ok": True}

    monkeypatch.setattr(
        "sora_agent.channels.slack.transport.request_json", fake_request
    )
    transport = UrllibSlackTransport("xoxb", "xapp")
    assert transport.post_message("C1", "hi", thread_ts="1.0") == "2.0"
    transport.set_status("C1", "1.0", "thinking")
    assert calls[0][0].endswith("chat.postMessage")
    assert calls[0][1]["json_body"]["thread_ts"] == "1.0"
    assert calls[1][1]["json_body"]["status"] == "thinking"


def test_transport_open_socket_and_auth(monkeypatch):
    replies = iter(
        [
            {"ok": True, "url": "wss://socket"},
            {"ok": True, "user_id": "BOT"},
        ]
    )

    def fake_request(url, **kwargs):
        return next(replies)

    monkeypatch.setattr(
        "sora_agent.channels.slack.transport.request_json", fake_request
    )
    transport = UrllibSlackTransport("xoxb", "xapp")
    assert transport.open_socket() == "wss://socket"
    assert transport.auth_test()["user_id"] == "BOT"


def test_transport_reports_slack_errors(monkeypatch):
    def fake_request(url, **kwargs):
        return {"ok": False, "error": "channel_not_found"}

    monkeypatch.setattr(
        "sora_agent.channels.slack.transport.request_json", fake_request
    )
    transport = UrllibSlackTransport("xoxb", "xapp")
    with pytest.raises(SlackError, match="channel_not_found"):
        transport.post_message("C1", "hi")
