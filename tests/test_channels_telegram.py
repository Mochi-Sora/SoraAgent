"""Tests for the Telegram channel and its standard-library transport."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from sora_agent.channels.approval import ApprovalBroker
from sora_agent.channels.bus import MessageBus
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.manager import ChannelManager
from sora_agent.channels.telegram.config import TelegramConfig, split_ids
from sora_agent.channels.telegram.runtime import (
    TelegramChannel,
    parse_update,
    split_text,
)
from sora_agent.channels.telegram.transport import (
    TelegramError,
    UrllibTelegramTransport,
)
from sora_agent.channels.types import InboundMessage, OutboundMessage
from sora_agent.files import FileUnderstander
from sora_agent.providers import ModelTurn, ToolCall
from sora_agent.tools import Tool, ToolRegistry

TOKEN = "123:abc"


def _config(**overrides) -> TelegramConfig:
    values = {"token": TOKEN, "allow_from": ("*",)}
    values.update(overrides)
    return TelegramConfig(**values)


def _update(text: str | None = "hello", sender=7, chat=42, update_id=1, **extra):
    message = {
        "message_id": update_id + 100,
        "from": {"id": sender},
        "chat": {"id": chat, "type": "private"},
        "text": text,
    }
    message.update(extra)
    return {"update_id": update_id, "message": message}


class FakeTransport:
    def __init__(self, updates=None, error=None) -> None:
        self.updates = list(updates or [])
        self.error = error
        self.offsets: list[int] = []
        self.sent: list[tuple] = []
        self.answered: list[tuple] = []
        self.cleared: list[tuple] = []
        self.actions: list[tuple] = []
        self.downloads: list[tuple] = []
        self.files: dict[str, bytes] = {}

    def get_updates(self, offset, timeout):
        self.offsets.append(offset)
        if self.error is not None:
            raise self.error
        if not self.updates:
            return []
        batch, self.updates = self.updates, []
        return batch

    def send_message(
        self,
        chat_id,
        text,
        *,
        reply_to=None,
        parse_mode=None,
        reply_markup=None,
        message_thread_id=None,
    ):
        self.sent.append(
            (
                chat_id,
                text,
                reply_to,
                parse_mode,
                reply_markup,
                message_thread_id,
            )
        )
        return "9"

    def send_chat_action(self, chat_id, action, *, message_thread_id=None):
        self.actions.append((chat_id, action, message_thread_id))

    def answer_callback_query(self, callback_query_id, text=""):
        self.answered.append((callback_query_id, text))

    def clear_reply_markup(self, chat_id, message_id):
        self.cleared.append((chat_id, message_id))

    def download_file(self, file_id, max_bytes):
        self.downloads.append((file_id, max_bytes))
        return self.files.get(file_id, b"attachment")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_requires_a_token():
    with pytest.raises(ValueError, match="SORA_TELEGRAM_TOKEN"):
        TelegramConfig.from_env({})


def test_config_from_env_reads_every_field():
    config = TelegramConfig.from_env(
        {
            "SORA_TELEGRAM_TOKEN": "t",
            "SORA_TELEGRAM_ALLOW_FROM": "1, 2",
            "SORA_TELEGRAM_POLL_TIMEOUT": "5",
            "SORA_TELEGRAM_MAX_CHARS": "1000",
            "SORA_TELEGRAM_PARSE_MODE": "HTML",
        }
    )
    assert config.allow_from == ("1", "2")
    assert config.poll_timeout == 5
    assert config.max_message_chars == 1000
    assert config.parse_mode == "HTML"


def test_config_rejects_unknown_parse_mode():
    with pytest.raises(ValueError, match="parse_mode"):
        _config(parse_mode="MarkdownV3")


def test_split_ids_drops_blanks():
    assert split_ids(" 1, ,2 , ") == ("1", "2")


# ---------------------------------------------------------------------------
# update parsing
# ---------------------------------------------------------------------------


def test_parse_update_maps_a_text_message():
    parsed = parse_update(_update("hi"))
    assert parsed is not None
    assert parsed.sender_id == "7"
    assert parsed.chat_id == "42"
    assert parsed.content == "hi"
    assert parsed.metadata["message_id"] == 101
    assert parsed.session_key is None


def test_parse_update_scopes_topics_to_their_thread():
    parsed = parse_update(_update("hi", message_thread_id=9))
    assert parsed is not None
    assert parsed.session_key == "telegram:42:9"


def test_parse_update_uses_the_largest_photo():
    parsed = parse_update(
        _update(
            text=None,
            caption="look",
            photo=[
                {"file_id": "small", "file_size": 10},
                {"file_id": "large", "file_size": 90},
            ],
        )
    )
    assert parsed is not None
    assert parsed.content == "look"
    assert parsed.media == ["file_id:large"]
    assert parsed.attachments[0].file_id == "large"
    assert parsed.attachments[0].kind == "image"
    assert parsed.attachments[0].mime_type == "image/jpeg"


def test_parse_update_describes_a_document_attachment():
    parsed = parse_update(
        _update(
            text=None,
            document={
                "file_id": "doc1",
                "file_name": "diagram.png",
                "mime_type": "image/png",
                "file_size": 1234,
            },
        )
    )

    assert parsed is not None
    attachment = parsed.attachments[0]
    assert attachment.filename == "diagram.png"
    assert attachment.kind == "image"
    assert attachment.size == 1234
    assert parsed.content.startswith("The user sent attachments")


def test_parse_update_ignores_edits_and_empty_messages():
    assert parse_update({"update_id": 1, "edited_message": {"text": "x"}}) is None
    assert parse_update({"update_id": 2, "message": {"chat": {"id": 1}}}) is None


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def test_split_text_prefers_word_boundaries():
    chunks = split_text("alpha beta gamma delta", 10)
    assert chunks == ["alpha", "beta gamma", "delta"]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_text_hard_splits_unbreakable_text():
    chunks = split_text("x" * 25, 10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_split_text_returns_nothing_for_blank_input():
    assert split_text("   ") == []


# ---------------------------------------------------------------------------
# channel runtime
# ---------------------------------------------------------------------------


def test_channel_publishes_allowed_updates():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport([_update("hi")])
        channel = TelegramChannel(_config(), bus, transport=transport)
        task = asyncio.create_task(channel.start())
        try:
            message = await asyncio.wait_for(bus.next_inbound(), 2)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return message, transport

    message, transport = asyncio.run(scenario())
    assert message.channel == "telegram"
    assert message.content == "hi"
    assert message.session_key == "telegram:42"
    assert transport.offsets[0] == 0


def test_channel_downloads_photo_attachments():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport(
            [
                _update(
                    text=None,
                    caption="look",
                    photo=[{"file_id": "large", "file_size": 7}],
                )
            ]
        )
        transport.files["large"] = b"jpeg"
        channel = TelegramChannel(_config(), bus, transport=transport)
        task = asyncio.create_task(channel.start())
        try:
            message = await asyncio.wait_for(bus.next_inbound(), 2)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return message, transport

    message, transport = asyncio.run(scenario())
    assert message.content == "look"
    assert message.media == ["file_id:large"]
    assert len(message.attachments) == 1
    attachment = message.attachments[0]
    assert attachment.data == b"jpeg"
    assert attachment.filename == "photo.jpg"
    assert attachment.kind == "image"
    assert attachment.size == 4
    assert transport.downloads == [("large", 20 * 1024 * 1024)]


def test_channel_marks_oversized_attachments_without_downloading():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport(
            [
                _update(
                    text=None,
                    document={
                        "file_id": "big",
                        "file_name": "big.pdf",
                        "mime_type": "application/pdf",
                        "file_size": 100,
                    },
                )
            ]
        )
        channel = TelegramChannel(
            _config(max_attachment_bytes=10), bus, transport=transport
        )
        task = asyncio.create_task(channel.start())
        try:
            message = await asyncio.wait_for(bus.next_inbound(), 2)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return message, transport

    message, transport = asyncio.run(scenario())
    assert "above the 10 byte Telegram limit" in message.attachments[0].error
    assert transport.downloads == []


def test_channel_records_download_failures():
    class FailingTransport(FakeTransport):
        def download_file(self, file_id, max_bytes):
            raise TelegramError("nope")

    async def scenario():
        bus = MessageBus()
        transport = FailingTransport([_update(text="see file", document={"file_id": "f"})])
        channel = TelegramChannel(_config(), bus, transport=transport)
        task = asyncio.create_task(channel.start())
        try:
            message = await asyncio.wait_for(bus.next_inbound(), 2)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return message

    message = asyncio.run(scenario())
    assert "download failed: nope" in message.attachments[0].error


def test_channel_drops_unlisted_senders():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport([_update("hi", sender=9)])
        channel = TelegramChannel(
            _config(allow_from=("7",)), bus, transport=transport
        )
        task = asyncio.create_task(channel.start())
        try:
            await asyncio.sleep(0.15)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return bus.inbound.qsize()

    assert asyncio.run(scenario()) == 0


def test_channel_does_not_download_from_unlisted_senders():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport(
            [
                _update(
                    text=None,
                    caption="look",
                    sender=9,
                    photo=[{"file_id": "secret", "file_size": 10}],
                )
            ]
        )
        channel = TelegramChannel(
            _config(allow_from=("7",)), bus, transport=transport
        )
        task = asyncio.create_task(channel.start())
        try:
            await asyncio.sleep(0.15)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return transport, bus.inbound.qsize()

    transport, queued = asyncio.run(scenario())
    assert transport.downloads == []
    assert queued == 0


def test_channel_advances_the_offset():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport([_update("a", update_id=5), _update("b", update_id=6)])
        channel = TelegramChannel(_config(), bus, transport=transport)
        task = asyncio.create_task(channel.start())
        try:
            await _wait_for(lambda: bus.inbound.qsize() == 2)
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return transport.offsets

    offsets = asyncio.run(scenario())
    assert offsets[0] == 0
    assert 7 in offsets


def test_send_chunks_long_messages():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport()
        channel = TelegramChannel(
            _config(max_message_chars=10), bus, transport=transport
        )
        await channel.send(
            OutboundMessage("telegram", "42", "alpha beta gamma delta")
        )
        return transport.sent

    sent = asyncio.run(scenario())
    assert len(sent) == 3
    assert all(len(text) <= 10 for _, text, _, _, _, _ in sent)
    assert (
        " ".join(text for _, text, _, _, _, _ in sent)
        == "alpha beta gamma delta"
    )
    assert sent[0][2] is None


def test_send_propagates_transport_errors():
    class FailingTransport(FakeTransport):
        def send_message(self, *args, **kwargs):
            raise TelegramError("boom")

    async def scenario():
        channel = TelegramChannel(
            _config(), MessageBus(), transport=FailingTransport()
        )
        with pytest.raises(TelegramError, match="boom"):
            await channel.send(OutboundMessage("telegram", "42", "hi"))

    asyncio.run(scenario())


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# urllib transport
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_transport_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(
            {"ok": False, "description": "bad token"}
        ),
    )
    transport = UrllibTelegramTransport(TOKEN)
    with pytest.raises(TelegramError, match="bad token"):
        transport.send_message("42", "hi")


def test_transport_sends_reply_and_parse_mode(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None, **kwargs):
        captured["request"] = request
        captured["context"] = kwargs.get("context")
        return _FakeResponse({"ok": True, "result": {}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    transport = UrllibTelegramTransport(TOKEN)
    transport.send_message(
        "42",
        "hi",
        reply_to="7",
        parse_mode="HTML",
        reply_markup={"inline_keyboard": []},
    )

    body = json.loads(captured["request"].data)
    assert body["chat_id"] == "42"
    assert body["reply_parameters"] == {"message_id": 7}
    assert body["parse_mode"] == "HTML"
    assert body["reply_markup"] == {"inline_keyboard": []}


def test_transport_rejects_oversized_files(monkeypatch):
    transport = UrllibTelegramTransport(TOKEN)
    monkeypatch.setattr(
        transport,
        "_call",
        lambda method, payload: {
            "file_path": "documents/big.bin",
            "file_size": 100,
        },
    )

    with pytest.raises(TelegramError, match="above the 10 byte limit"):
        transport.download_file("f1", 10)


def test_transport_answers_callback_queries(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None, **kwargs):
        captured["request"] = request
        return _FakeResponse({"ok": True, "result": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    transport = UrllibTelegramTransport(TOKEN)
    transport.answer_callback_query("cq1", "Approved")

    assert captured["request"].full_url.endswith("/answerCallbackQuery")
    body = json.loads(captured["request"].data)
    assert body == {"callback_query_id": "cq1", "text": "Approved"}


class _BotApiHandler(BaseHTTPRequestHandler):
    captured: ClassVar[list] = []
    queued: ClassVar[list] = []
    downloads: ClassVar[dict[str, bytes]] = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).captured.append((self.path, body))
        response = (
            type(self).queued.pop(0)
            if type(self).queued
            else {"ok": True, "result": []}
        )
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        type(self).captured.append((self.path, None))
        raw = type(self).downloads.get(self.path, b"file-bytes")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args, **kwargs):
        return


def test_transport_round_trips_over_real_http():
    _BotApiHandler.captured = []
    _BotApiHandler.queued = [
        {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 2,
                        "from": {"id": 7},
                        "chat": {"id": 42},
                        "text": "hi",
                    },
                }
            ],
        },
        {"ok": True, "result": {"message_id": 3}},
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BotApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        transport = UrllibTelegramTransport(
            "tok", f"http://{host}:{port}", request_timeout=2.0
        )
        updates = transport.get_updates(0, 0)
        transport.send_message("42", "hello", reply_to="2")
    finally:
        server.shutdown()
        server.server_close()

    assert updates[0]["update_id"] == 1
    paths = [path for path, _ in _BotApiHandler.captured]
    assert paths == ["/bottok/getUpdates", "/bottok/sendMessage"]
    assert _BotApiHandler.captured[0][1]["allowed_updates"] == [
        "message",
        "callback_query",
    ]
    assert _BotApiHandler.captured[1][1]["reply_parameters"] == {"message_id": 2}


def test_transport_downloads_files_over_real_http():
    _BotApiHandler.captured = []
    _BotApiHandler.downloads = {"/file/bottok/documents/a.txt": b"file bytes"}
    _BotApiHandler.queued = [
        {
            "ok": True,
            "result": {
                "file_id": "f1",
                "file_path": "documents/a.txt",
                "file_size": 10,
            },
        }
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BotApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        transport = UrllibTelegramTransport(
            "tok", f"http://{host}:{port}", request_timeout=2.0
        )
        data = transport.download_file("f1", 100)
    finally:
        server.shutdown()
        server.server_close()

    assert data == b"file bytes"
    assert [path for path, _ in _BotApiHandler.captured] == [
        "/bottok/getFile",
        "/file/bottok/documents/a.txt",
    ]


# ---------------------------------------------------------------------------
# approval buttons
# ---------------------------------------------------------------------------


def _callback_update(data: str = "approval:yes:abc") -> dict:
    return {
        "update_id": 9,
        "callback_query": {
            "id": "cq1",
            "from": {"id": 7},
            "data": data,
            "message": {
                "message_id": 10,
                "chat": {"id": 42, "type": "private"},
            },
        },
    }


def test_parse_update_maps_an_approval_callback():
    parsed = parse_update(_callback_update())
    assert parsed is not None
    assert parsed.content == "yes"
    assert parsed.metadata["kind"] == "approval_callback"
    assert parsed.metadata["approval_id"] == "abc"
    assert parsed.callback_query_id == "cq1"
    assert parsed.callback_message_id == "10"
    assert parsed.session_key is None


def test_parse_update_ignores_unknown_callback_data():
    assert parse_update(_callback_update("nope:yes:abc")) is None
    assert parse_update(_callback_update("approval:maybe:abc")) is None


def test_send_renders_approval_buttons():
    async def scenario():
        transport = FakeTransport()
        channel = TelegramChannel(_config(), MessageBus(), transport=transport)
        await channel.send(
            OutboundMessage(
                "telegram",
                "42",
                "Approve?",
                metadata={"kind": "approval", "approval_id": "abc"},
                buttons=[["Approve", "Reject"]],
            )
        )
        return transport.sent

    sent = asyncio.run(scenario())
    assert sent[0][4] == {
        "inline_keyboard": [
            [
                {"text": "Approve", "callback_data": "approval:yes:abc"},
                {"text": "Reject", "callback_data": "approval:no:abc"},
            ]
        ]
    }


def test_channel_sends_typing_action():
    async def scenario():
        transport = FakeTransport()
        channel = TelegramChannel(_config(), MessageBus(), transport=transport)
        inbound = InboundMessage(
            "telegram",
            "7",
            "42",
            "hi",
            metadata={"message_thread_id": 9},
        )
        await channel.send_typing(inbound)
        return transport

    transport = asyncio.run(scenario())
    assert transport.actions == [("42", "typing", 9)]


def test_send_forwards_topic_thread_id():
    async def scenario():
        transport = FakeTransport()
        channel = TelegramChannel(_config(), MessageBus(), transport=transport)
        await channel.send(
            OutboundMessage(
                "telegram",
                "42",
                "hello",
                metadata={"message_thread_id": 9},
            )
        )
        return transport.sent

    sent = asyncio.run(scenario())
    assert sent[0][5] == 9


def test_send_without_approval_metadata_has_no_buttons():
    async def scenario():
        transport = FakeTransport()
        channel = TelegramChannel(_config(), MessageBus(), transport=transport)
        await channel.send(
            OutboundMessage(
                "telegram", "42", "hello", buttons=[["Approve", "Reject"]]
            )
        )
        return transport.sent

    assert asyncio.run(scenario())[0][4] is None


def test_channel_acknowledges_button_presses():
    async def scenario():
        bus = MessageBus()
        transport = FakeTransport([_callback_update()])
        channel = TelegramChannel(_config(), bus, transport=transport)
        task = asyncio.create_task(channel.start())
        try:
            message = await asyncio.wait_for(bus.next_inbound(), 2)
            await _wait_for(lambda: bool(transport.answered))
        finally:
            await channel.stop()
            await asyncio.wait_for(task, 2)
        return message, transport

    message, transport = asyncio.run(scenario())
    assert message.content == "yes"
    assert transport.answered == [("cq1", "Approved")]
    assert transport.cleared == [("42", "10")]


class CallbackTransport(FakeTransport):
    """Serve one button press after the approval prompt is delivered."""

    def __init__(self, first_update) -> None:
        super().__init__([first_update])
        self.pending_callback = True

    def send_message(
        self,
        chat_id,
        text,
        *,
        reply_to=None,
        parse_mode=None,
        reply_markup=None,
        message_thread_id=None,
    ):
        message_id = super().send_message(
            chat_id,
            text,
            reply_to=reply_to,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            message_thread_id=message_thread_id,
        )
        if not reply_markup or not self.pending_callback:
            return message_id
        self.pending_callback = False
        data = reply_markup["inline_keyboard"][0][0]["callback_data"]
        self.updates.append(_callback_update(data))
        return message_id


def test_attachment_reaches_the_system_prompt_through_the_manager():
    class RecordingProvider:
        def __init__(self):
            self.calls = []

        def complete_turn(self, messages, tools):
            self.calls.append(list(messages))
            return ModelTurn(text="done")

    provider = RecordingProvider()

    async def scenario():
        bus = MessageBus()
        handler = ChatHandler(
            provider,
            max_steps=1,
            clock=False,
            quality=False,
            file_understander=FileUnderstander(),
        )
        transport = FakeTransport(
            [
                _update(
                    text="read this",
                    document={
                        "file_id": "doc",
                        "file_name": "notes.txt",
                        "mime_type": "text/plain",
                        "file_size": 13,
                    },
                )
            ]
        )
        transport.files["doc"] = b"file contents"
        channel = TelegramChannel(_config(), bus, transport=transport)
        manager = ChannelManager([channel], bus, handler.handle)
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(
                lambda: any(sent[1] == "done" for sent in transport.sent)
            )
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return transport

    transport = asyncio.run(scenario())
    system = provider.calls[-1][0].content
    assert "## Attached files (dynamic)" in system
    assert "file contents" in system
    assert any(sent[1] == "done" for sent in transport.sent)


def test_button_press_completes_an_approval():
    executed: list[str] = []

    def expedite(order_id: str) -> str:
        executed.append(order_id)
        return "expedited"

    registry = ToolRegistry()
    registry.add(
        Tool(expedite, "expedite_order", "Expedite.", needs_approval=True)
    )

    class Provider:
        def complete_turn(self, messages, tools):
            if any(message.role == "tool" for message in messages):
                return ModelTurn(text="done")
            return ModelTurn(
                tool_calls=[
                    ToolCall("c1", "expedite_order", {"order_id": "PO-1"})
                ]
            )

    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        handler = ChatHandler(
            Provider(), registry, approvals=broker, max_steps=2
        )
        transport = CallbackTransport(_update("go"))
        channel = TelegramChannel(_config(), bus, transport=transport)
        manager = ChannelManager(
            [channel], bus, handler.handle, approvals=broker
        )
        task = asyncio.create_task(manager.serve())
        try:
            await _wait_for(
                lambda: any(sent[1] == "done" for sent in transport.sent)
            )
        finally:
            manager.request_stop()
            await asyncio.wait_for(task, 3)
        return transport

    transport = asyncio.run(scenario())
    assert executed == ["PO-1"]
    assert transport.answered == [("cq1", "Approved")]
    assert any(markup for _, _, _, _, markup, _ in transport.sent)
