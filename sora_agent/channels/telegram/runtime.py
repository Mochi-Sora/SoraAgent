"""Telegram channel runtime: long polling in, chunked replies out."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from dataclasses import dataclass, field
from typing import Any

from ...files import Attachment
from ..approval import APPROVAL_CALLBACK_KIND, APPROVAL_KIND
from ..base import BaseChannel
from ..bus import MessageBus
from ..types import InboundMessage, OutboundMessage
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
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_key: str | None = None
    callback_query_id: str | None = None
    callback_message_id: str | None = None


class TelegramChannel(BaseChannel):
    """Long-polls the Bot API and delivers replies in chunks."""

    name = "telegram"
    display_name = "Telegram"

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
        chunks = split_text(msg.content, self.config.max_message_chars)
        markup = _approval_markup(msg)
        thread_id = msg.metadata.get("message_thread_id")
        for index, chunk in enumerate(chunks):
            await asyncio.to_thread(
                self.transport.send_message,
                msg.chat_id,
                chunk,
                reply_to=msg.reply_to if index == 0 else None,
                parse_mode=self.config.parse_mode,
                reply_markup=markup if index == 0 else None,
                message_thread_id=thread_id,
            )

    async def send_typing(self, inbound: InboundMessage) -> None:
        await asyncio.to_thread(
            self.transport.send_chat_action,
            inbound.chat_id,
            "typing",
            message_thread_id=inbound.metadata.get("message_thread_id"),
        )

    async def _dispatch_update(self, update: dict[str, Any]) -> None:
        parsed = parse_update(update)
        if parsed is None:
            return
        if parsed.attachments and not self.is_allowed(parsed.sender_id):
            self.log.debug(
                "ignored attachments from unauthorized sender %s",
                parsed.sender_id,
            )
            return
        attachments = await self._download_attachments(parsed.attachments)
        accepted = await self._handle_message(
            sender_id=parsed.sender_id,
            chat_id=parsed.chat_id,
            content=parsed.content,
            media=parsed.media,
            attachments=attachments,
            metadata=parsed.metadata,
            session_key=parsed.session_key,
        )
        if parsed.callback_query_id:
            await self._acknowledge_callback(parsed)
        if accepted:
            self.log.debug("queued telegram message from %s", parsed.sender_id)

    async def _download_attachments(
        self, attachments: list[Attachment]
    ) -> list[Attachment]:
        if not attachments:
            return attachments
        limit = self.config.max_attachment_bytes
        if not self.config.download_attachments:
            for attachment in attachments:
                attachment.error = "attachment downloads are disabled"
            return attachments
        for attachment in attachments:
            if not attachment.file_id:
                continue
            if attachment.size and attachment.size > limit:
                attachment.error = (
                    f"file is {attachment.size} bytes, above the {limit} "
                    "byte Telegram limit"
                )
                continue
            try:
                data = await asyncio.to_thread(
                    self.transport.download_file,
                    attachment.file_id,
                    limit,
                )
            except Exception as exc:  # noqa: BLE001 - keep the turn alive
                attachment.error = f"download failed: {exc}"
                continue
            attachment.data = data
            attachment.size = len(data)
        return attachments

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
    attachments = _attachments(message)
    media = [
        f"file_id:{attachment.file_id}"
        for attachment in attachments
        if attachment.file_id
    ]
    if not content and attachments:
        content = "The user sent attachments without any text."
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
        attachments=attachments,
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


def _attachments(message: dict[str, Any]) -> list[Attachment]:
    """Describe every attachment in a Telegram message."""
    attachments: list[Attachment] = []
    photos = message.get("photo")
    if isinstance(photos, list):
        candidates = [item for item in photos if isinstance(item, dict)]
        if candidates:
            largest = max(
                candidates, key=lambda item: _int_value(item.get("file_size"))
            )
            if largest.get("file_id"):
                attachments.append(
                    Attachment(
                        filename="photo.jpg",
                        mime_type="image/jpeg",
                        kind="image",
                        file_id=str(largest["file_id"]),
                        size=_int_value(largest.get("file_size")),
                        metadata={
                            "width": _int_value(largest.get("width")),
                            "height": _int_value(largest.get("height")),
                        },
                    )
                )

    document = message.get("document")
    if isinstance(document, dict) and document.get("file_id"):
        filename = str(document.get("file_name") or "document.bin")
        mime = str(document.get("mime_type") or "")
        attachments.append(
            Attachment(
                filename=filename,
                mime_type=mime,
                kind=_kind_from_mime(mime, filename),
                file_id=str(document["file_id"]),
                size=_int_value(document.get("file_size")),
            )
        )

    specs = (
        ("voice", "voice.ogg", "audio", "audio/ogg"),
        ("audio", "audio.mp3", "audio", "audio/mpeg"),
        ("video", "video.mp4", "video", "video/mp4"),
        ("video_note", "video_note.mp4", "video", "video/mp4"),
        ("animation", "animation.mp4", "video", "video/mp4"),
    )
    for key, fallback_name, kind, fallback_mime in specs:
        item = message.get(key)
        if not isinstance(item, dict) or not item.get("file_id"):
            continue
        mime = str(item.get("mime_type") or fallback_mime)
        filename = str(item.get("file_name") or fallback_name)
        metadata = {
            "duration": item.get("duration"),
            "width": item.get("width"),
            "height": item.get("height"),
            "title": item.get("title"),
            "performer": item.get("performer"),
        }
        metadata = {
            name: value
            for name, value in metadata.items()
            if value not in (None, "")
        }
        attachments.append(
            Attachment(
                filename=filename,
                mime_type=mime,
                kind=kind,
                file_id=str(item["file_id"]),
                size=_int_value(item.get("file_size")),
                metadata=metadata,
            )
        )

    sticker = message.get("sticker")
    if isinstance(sticker, dict) and sticker.get("file_id"):
        video = bool(sticker.get("is_video"))
        attachments.append(
            Attachment(
                filename="sticker.webm" if video else "sticker.webp",
                mime_type="video/webm" if video else "image/webp",
                kind="video" if video else "image",
                file_id=str(sticker["file_id"]),
                size=_int_value(sticker.get("file_size")),
                metadata={
                    "width": sticker.get("width"),
                    "height": sticker.get("height"),
                    "emoji": sticker.get("emoji"),
                },
            )
        )
    return attachments


def _kind_from_mime(mime: str, filename: str) -> str:
    value = (mime or mimetypes.guess_type(filename)[0] or "").lower()
    if value.startswith("image/"):
        return "image"
    if value.startswith("audio/"):
        return "audio"
    if value.startswith("video/"):
        return "video"
    return "document"


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _advance_offset(offset: int, update: dict[str, Any]) -> int:
    raw = update.get("update_id")
    if raw is None:
        return offset
    try:
        update_id = int(raw)
    except (TypeError, ValueError):
        return offset
    return max(offset, update_id + 1)
