"""Matrix channel runtime using the Client-Server API."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from ...files import Attachment
from .._util import Dedupe, split_text
from ..base import BaseChannel
from ..types import InboundMessage, OutboundMessage
from .config import MatrixConfig
from .transport import MatrixError, MatrixTransport, UrllibMatrixTransport


class MatrixChannel(BaseChannel):
    """A Matrix bot using ``/sync``; end-to-end encryption is not supported."""

    name = "matrix"
    display_name = "Matrix"

    def __init__(
        self,
        config: MatrixConfig,
        bus,
        *,
        transport: MatrixTransport | None = None,
        sleeper=time.sleep,
    ) -> None:
        super().__init__(config, bus)
        self.config: MatrixConfig = config
        self.transport = transport or UrllibMatrixTransport(
            config.homeserver,
            config.access_token,
            request_timeout=config.request_timeout,
        )
        self._sleeper = sleeper
        self._dedupe = Dedupe()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._user_id = config.user_id.strip()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        if not self._user_id:
            try:
                self._user_id = self.transport.whoami()
            except Exception as exc:  # noqa: BLE001 - sync will report auth errors
                self.log.warning("matrix whoami failed: %s", exc)
        self._thread = threading.Thread(
            target=self._sync_loop,
            name="matrix-sync",
            daemon=True,
        )
        self._thread.start()
        try:
            await self._stopped.wait()
        finally:
            thread = self._thread
            self._thread = None
            if thread is not None:
                thread.join(timeout=5)

    async def stop(self) -> None:
        self._running = False
        loop = self._loop
        stopped = self._stopped
        if loop is not None and stopped is not None:
            loop.call_soon_threadsafe(stopped.set)

    async def send(self, msg: OutboundMessage) -> None:
        thread_root = msg.metadata.get("thread_root")
        for index, chunk in enumerate(
            split_text(msg.content, self.config.max_message_chars)
        ):
            self.transport.send_message(
                msg.chat_id,
                chunk,
                thread_root=str(thread_root) if thread_root else None,
                reply_to=msg.reply_to if index == 0 else None,
            )

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing or not self._user_id:
            return
        try:
            self.transport.send_typing(inbound.chat_id, self._user_id)
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("matrix typing indicator failed: %s", exc)

    # ------------------------------------------------------------------
    # sync thread
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        since: str | None = None
        try:
            try:
                initial = self.transport.sync(None, 0)
                since = _next_batch(initial)
            except Exception as exc:  # noqa: BLE001 - retry below
                self.log.warning("matrix initial sync failed: %s", exc)
            while self._running:
                try:
                    payload = self.transport.sync(
                        since,
                        self.config.sync_timeout,
                    )
                    since = _next_batch(payload) or since
                    for message in parse_sync(payload, self._user_id):
                        self._ingest(message)
                except MatrixError as exc:
                    if not self._running:
                        break
                    self.log.warning("matrix sync failed: %s", exc)
                    self._sleeper(self.config.reconnect_delay)
                except Exception as exc:  # noqa: BLE001 - retry on transport errors
                    if not self._running:
                        break
                    self.log.warning("matrix sync error: %s", exc)
                    self._sleeper(self.config.reconnect_delay)
        finally:
            self._running = False
            loop = self._loop
            stopped = self._stopped
            if loop is not None and stopped is not None:
                loop.call_soon_threadsafe(stopped.set)

    def _ingest(self, message: InboundMessage) -> None:
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate matrix message")
            return
        self._prepare_attachments(message)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(
            self.bus.publish_inbound(message), loop
        )
        try:
            future.result(timeout=10)
        except Exception:  # one bad message must not stop sync
            self.log.exception("failed to publish matrix message")

    def _prepare_attachments(self, message: InboundMessage) -> None:
        if not self.config.download_attachments:
            return
        for attachment in message.attachments:
            if attachment.data is not None or attachment.error:
                continue
            if attachment.size and attachment.size > self.config.max_attachment_bytes:
                attachment.error = (
                    f"attachment is above the "
                    f"{self.config.max_attachment_bytes} byte limit"
                )
                continue
            try:
                attachment.data = self.transport.download_media(
                    attachment.file_id,
                    self.config.max_attachment_bytes,
                )
                attachment.size = len(attachment.data)
            except Exception as exc:  # noqa: BLE001 - report on the attachment
                attachment.error = f"download failed: {exc}"


def parse_sync(payload: Any, user_id: str = "") -> list[InboundMessage]:
    """Translate the joined-room timeline of one ``/sync`` response."""
    if not isinstance(payload, dict):
        return []
    rooms = payload.get("rooms")
    joined = rooms.get("join") if isinstance(rooms, dict) else None
    if not isinstance(joined, dict):
        return []
    result: list[InboundMessage] = []
    for room_id, room in joined.items():
        if not isinstance(room, dict):
            continue
        timeline = room.get("timeline")
        events = timeline.get("events") if isinstance(timeline, dict) else None
        if not isinstance(events, list):
            continue
        for event in events:
            message = parse_event(event, str(room_id), user_id)
            if message is not None:
                result.append(message)
    return result


def parse_event(
    event: Any,
    room_id: str,
    user_id: str = "",
) -> InboundMessage | None:
    """Translate one Matrix room event."""
    if not isinstance(event, dict):
        return None
    if event.get("type") != "m.room.message":
        return None
    sender = event.get("sender")
    if sender is None or (user_id and str(sender) == str(user_id)):
        return None
    content = event.get("content")
    if not isinstance(content, dict):
        return None
    relates = content.get("m.relates_to")
    if isinstance(relates, dict) and relates.get("rel_type") == "m.replace":
        return None
    msgtype = str(content.get("msgtype") or "")
    body = content.get("body")
    body = body if isinstance(body, str) else ""
    attachments = _attachments(content, msgtype)
    if not body.strip() and not attachments:
        return None
    if not body.strip() and attachments:
        body = "The user sent attachments."
    thread_root = None
    if isinstance(relates, dict) and relates.get("rel_type") == "m.thread":
        thread_root = relates.get("event_id")
    event_id = event.get("event_id")
    metadata: dict[str, Any] = {
        "message_id": event_id,
        "room_id": room_id,
        "thread_root": thread_root,
        "sender": str(sender),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    session_key = (
        f"matrix:{room_id}:{thread_root}"
        if thread_root
        else f"matrix:{room_id}"
    )
    return InboundMessage(
        channel="matrix",
        sender_id=str(sender),
        chat_id=room_id,
        content=body,
        media=[item.file_id for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=session_key,
    )


def _attachments(content: dict[str, Any], msgtype: str) -> list[Attachment]:
    if msgtype not in {"m.image", "m.file", "m.audio", "m.video"}:
        return []
    info = content.get("info")
    info = info if isinstance(info, dict) else {}
    mime_type = str(info.get("mimetype") or "")
    filename = str(content.get("body") or "attachment")
    if content.get("file"):
        return [
            Attachment(
                filename=filename,
                mime_type=mime_type,
                kind=_kind_from_mime(mime_type),
                error="encrypted Matrix attachments are not supported",
            )
        ]
    url = content.get("url")
    if not url:
        return []
    return [
        Attachment(
            filename=filename,
            mime_type=mime_type,
            kind=_kind_from_mime(mime_type),
            file_id=str(url),
            size=int(info.get("size") or 0),
        )
    ]


def _kind_from_mime(mime_type: str) -> str:
    mime = (mime_type or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("text/"):
        return "text"
    return "document"


def _next_batch(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("next_batch")
    return None if value is None else str(value)
