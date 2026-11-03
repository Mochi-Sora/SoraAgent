"""Telegram channel runtime: long polling in, chunked replies out."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..approval import APPROVAL_CALLBACK_KIND, APPROVAL_KIND
from ..base import BaseChannel
from ..bus import MessageBus
from ..types import (
    STREAM_DELTA_KIND,
    STREAM_END_KIND,
    OutboundMessage,
)
from .config import TelegramConfig
from .transport import TelegramTransport, UrllibTelegramTransport

_IDLE_DELAY = 0.05
_MAX_BACKOFF = 30.0


@dataclass
class ParsedUpdate:
    """One Telegram update reduced to what the agent needs."""

    sender_id: str
    chat_id: str
    content: str
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_key: str | None = None
    callback_query_id: str | None = None
    callback_message_id: str | None = None


class TelegramChannel(BaseChannel):
    """Long-polls the Bot API and delivers replies in chunks."""

    name = "telegram"
    display_name = "Telegram"
    supports_streaming = True

    def __init__(
        self,
        config: TelegramConfig,
        bus: MessageBus,
        *,
        transport: TelegramTransport | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config, bus)
        self.config = config
        if logger is not None:
            self.log = logger
        self.transport: TelegramTransport = transport or UrllibTelegramTransport(
            config.token,
            config.api_base,
            request_timeout=config.request_timeout,
        )
        self._streams: dict[str, str] = {}
        self._stream_text: dict[str, str] = {}

    async def start(self) -> None:
        """Poll until :meth:`stop` is called.

        A poll already in flight finishes first; the manager cancels the task
        after ``stop()`` so shutdown never waits a full poll timeout.
        """
        self._running = True
        offset = 0
        failures = 0
        self.log.info("telegram polling started")
        while self._running:
            try:
                updates = await asyncio.to_thread(
                    self.transport.get_updates, offset, self.config.poll_timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                if not self._running:
                    break
                failures += 1
                delay = min(_MAX_BACKOFF, 2.0 ** min(failures, 5))
                self.log.warning(
                    "telegram poll failed (%d): %s; retrying in %.0fs",
                    failures,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            failures = 0
            for update in updates:
                offset = _advance_offset(offset, update)
                await self._dispatch_update(update)
            if not updates:
                await asyncio.sleep(_IDLE_DELAY)
        self.log.info("telegram polling stopped")

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        kind = msg.metadata.get("kind")
        if kind == STREAM_END_KIND:
            stream_id = str(msg.metadata.get("stream_id") or "")
            self._streams.pop(stream_id, None)
            self._stream_text.pop(stream_id, None)
            return
        if kind == STREAM_DELTA_KIND:
            await self._send_delta(msg)
            return
        chunks = split_text(msg.content, self.config.max_message_chars)
        markup = _approval_markup(msg)
        for index, chunk in enumerate(chunks):
            await asyncio.to_thread(
                self.transport.send_message,
                msg.chat_id,
                chunk,
                reply_to=msg.reply_to if index == 0 else None,
                parse_mode=self.config.parse_mode,
                reply_markup=markup if index == 0 else None,
            )

    async def _send_delta(self, msg: OutboundMessage) -> None:
        """Keep one live message per stream and edit it as text grows."""
        stream_id = str(msg.metadata.get("stream_id") or "")
        text = msg.content
        if not stream_id or not text.strip():
            return
        if len(text) > self.config.max_message_chars:
            # Too large to edit live; the final message is chunked normally.
            return
        if self._stream_text.get(stream_id) == text:
            return
        message_id = self._streams.get(stream_id)
        if not message_id:
            sent_id = await asyncio.to_thread(
                self.transport.send_message,
                msg.chat_id,
                text,
                reply_to=msg.reply_to,
                parse_mode=self.config.parse_mode,
            )
            self._streams[stream_id] = sent_id or ""
            self._stream_text[stream_id] = text
            return
        await asyncio.to_thread(
            self.transport.edit_message_text,
            msg.chat_id,
            message_id,
            text,
            parse_mode=self.config.parse_mode,
        )
        self._stream_text[stream_id] = text

    async def _dispatch_update(self, update: dict[str, Any]) -> None:
        parsed = parse_update(update)
        if parsed is None:
            return
        accepted = await self._handle_message(
            sender_id=parsed.sender_id,
            chat_id=parsed.chat_id,
            content=parsed.content,
            media=parsed.media,
            metadata=parsed.metadata,
            session_key=parsed.session_key,
        )
        if parsed.callback_query_id:
            await self._acknowledge_callback(parsed)
        if accepted:
            self.log.debug("queued telegram message from %s", parsed.sender_id)

    async def _acknowledge_callback(self, parsed: ParsedUpdate) -> None:
        """Stop the button spinner and drop the keyboard.  Best effort."""
        query_id = parsed.callback_query_id
        if not query_id:
            return
        text = "Approved" if parsed.content == "yes" else "Rejected"
        try:
            await asyncio.to_thread(
                self.transport.answer_callback_query, query_id, text
            )
        except Exception as exc:  # noqa: BLE001 - ack is cosmetic
            self.log.debug("telegram callback acknowledgement failed: %s", exc)
        if not parsed.callback_message_id:
            return
        try:
            await asyncio.to_thread(
                self.transport.clear_reply_markup,
                parsed.chat_id,
                parsed.callback_message_id,
            )
        except Exception as exc:  # noqa: BLE001 - cleanup is cosmetic
            self.log.debug("telegram markup cleanup failed: %s", exc)


def parse_update(update: dict[str, Any]) -> ParsedUpdate | None:
    """Return the message or button press an update carries.

    Edited messages are ignored on purpose: replaying an edit would produce a
    second answer to the same question.
    """
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        return _parse_callback(update, callback)
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict) or chat.get("id") is None:
        return None
    chat_id = str(chat["id"])

    sender = message.get("from")
    if isinstance(sender, dict) and sender.get("id") is not None:
        sender_id = str(sender["id"])
    else:
        sender_id = chat_id

    content = message.get("text") or message.get("caption") or ""
    media = _media_references(message)
    if not content and not media:
        return None

    metadata: dict[str, Any] = {
        "message_id": message.get("message_id"),
        "update_id": update.get("update_id"),
        "chat_type": chat.get("type"),
    }
    thread_id = message.get("message_thread_id")
    session_key = None
    if thread_id is not None:
        metadata["message_thread_id"] = thread_id
        session_key = f"telegram:{chat_id}:{thread_id}"
    return ParsedUpdate(
        sender_id=sender_id,
        chat_id=chat_id,
        content=str(content),
        media=media,
        metadata=metadata,
        session_key=session_key,
    )


def _parse_callback(
    update: dict[str, Any], callback: dict[str, Any]
) -> ParsedUpdate | None:
    data = _parse_callback_data(str(callback.get("data") or ""))
    if data is None:
        return None
    decision, approval_id = data
    message = callback.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict) or chat.get("id") is None:
        return None
    chat_id = str(chat["id"])

    sender = callback.get("from")
    if isinstance(sender, dict) and sender.get("id") is not None:
        sender_id = str(sender["id"])
    else:
        sender_id = chat_id

    query_id = str(callback.get("id") or "")
    message_id = message.get("message_id")
    metadata: dict[str, Any] = {
        "kind": APPROVAL_CALLBACK_KIND,
        "approval_id": approval_id,
        "callback_query_id": query_id,
        "message_id": message_id,
        "update_id": update.get("update_id"),
        "chat_type": chat.get("type"),
    }
    thread_id = message.get("message_thread_id")
    session_key = None
    if thread_id is not None:
        metadata["message_thread_id"] = thread_id
        session_key = f"telegram:{chat_id}:{thread_id}"
    return ParsedUpdate(
        sender_id=sender_id,
        chat_id=chat_id,
        content=decision,
        metadata=metadata,
        session_key=session_key,
        callback_query_id=query_id or None,
        callback_message_id=None if message_id is None else str(message_id),
    )


def _parse_callback_data(data: str) -> tuple[str, str] | None:
    """Decode ``approval:yes:<id>`` into a decision and its token."""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "approval":
        return None
    if parts[1] not in {"yes", "no"} or not parts[2]:
        return None
    return parts[1], parts[2]


def _approval_markup(msg: OutboundMessage) -> dict[str, Any] | None:
    """Render approve/reject buttons for an approval prompt."""
    if msg.metadata.get("kind") != APPROVAL_KIND:
        return None
    approval_id = str(msg.metadata.get("approval_id") or "")
    if not approval_id:
        return None
    labels = msg.buttons[0] if msg.buttons else []
    if len(labels) != 2:
        return None
    return {
        "inline_keyboard": [
            [
                {
                    "text": str(labels[0]),
                    "callback_data": f"approval:yes:{approval_id}",
                },
                {
                    "text": str(labels[1]),
                    "callback_data": f"approval:no:{approval_id}",
                },
            ]
        ]
    }


def split_text(text: str, limit: int = 4096) -> list[str]:
    """Split ``text`` into platform-sized chunks, preferring natural breaks."""
    if limit <= 0:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = _break_point(remaining[:limit], limit)
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    tail = remaining.strip()
    if tail:
        chunks.append(tail)
    return chunks


def _break_point(window: str, limit: int) -> int:
    floor = max(1, limit // 2)
    for separator in ("\n\n", "\n", " "):
        index = window.rfind(separator)
        if index >= floor:
            return index
    return limit


def _media_references(message: dict[str, Any]) -> list[str]:
    """Record file ids for attachments without downloading them."""
    refs: list[str] = []
    photos = message.get("photo")
    if isinstance(photos, list):
        candidates = [item for item in photos if isinstance(item, dict)]
        if candidates:
            largest = max(
                candidates, key=lambda item: item.get("file_size") or 0
            )
            if largest.get("file_id"):
                refs.append(f"file_id:{largest['file_id']}")
    for key in ("document", "voice", "audio", "video", "video_note"):
        item = message.get(key)
        if isinstance(item, dict) and item.get("file_id"):
            refs.append(f"file_id:{item['file_id']}")
    return refs


def _advance_offset(offset: int, update: dict[str, Any]) -> int:
    raw = update.get("update_id")
    if raw is None:
        return offset
    try:
        update_id = int(raw)
    except (TypeError, ValueError):
        return offset
    return max(offset, update_id + 1)
