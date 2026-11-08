"""Signal channel runtime backed by a signal-cli daemon."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from ...files import Attachment
from .._sse import iter_sse
from .._util import Dedupe, split_text
from ..base import BaseChannel
from ..types import InboundMessage, OutboundMessage
from .config import SignalConfig
from .transport import SignalTransport, UrllibSignalTransport


class SignalChannel(BaseChannel):
    """A Signal bot connected to ``signal-cli`` over HTTP/SSE."""

    name = "signal"
    display_name = "Signal"

    def __init__(
        self,
        config: SignalConfig,
        bus,
        *,
        transport: SignalTransport | None = None,
        sleeper=time.sleep,
    ) -> None:
        super().__init__(config, bus)
        self.config: SignalConfig = config
        self.transport = transport or UrllibSignalTransport(
            config.http_url,
            config.account,
            request_timeout=config.request_timeout,
        )
        self._sleeper = sleeper
        self._dedupe = Dedupe()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        self._thread = threading.Thread(
            target=self._signal_loop,
            name="signal-events",
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
        try:
            self.transport.close()
        except Exception as exc:  # noqa: BLE001 - closing is best effort
            self.log.debug("signal event stream close failed: %s", exc)
        loop = self._loop
        stopped = self._stopped
        if loop is not None and stopped is not None:
            loop.call_soon_threadsafe(stopped.set)

    async def send(self, msg: OutboundMessage) -> None:
        for chunk in split_text(msg.content, self.config.max_message_chars):
            self.transport.send_text(self.config.account, msg.chat_id, chunk)

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing:
            return
        try:
            self.transport.send_typing(self.config.account, inbound.chat_id)
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("signal typing indicator failed: %s", exc)

    # ------------------------------------------------------------------
    # event thread
    # ------------------------------------------------------------------

    def _signal_loop(self) -> None:
        try:
            while self._running:
                try:
                    for event in iter_sse(self.transport.events()):
                        if not self._running:
                            break
                        self._dispatch(event)
                except Exception as exc:  # noqa: BLE001 - reconnect on stream errors
                    if not self._running:
                        break
                    self.log.warning("signal event stream failed: %s", exc)
                if not self._running:
                    break
                self._sleeper(self.config.reconnect_delay)
        finally:
            self._running = False
            loop = self._loop
            stopped = self._stopped
            if loop is not None and stopped is not None:
                loop.call_soon_threadsafe(stopped.set)

    def _dispatch(self, event: Any) -> None:
        message = parse_event(event)
        if message is None:
            return
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate signal message")
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(
            self.bus.publish_inbound(message), loop
        )
        try:
            future.result(timeout=10)
        except Exception:
            self.log.exception("failed to publish signal message")


def parse_event(event: Any) -> InboundMessage | None:
    """Translate one signal-cli receive event."""
    if not isinstance(event, dict):
        return None
    params = event.get("params")
    envelope = params.get("envelope") if isinstance(params, dict) else None
    if not isinstance(envelope, dict):
        envelope = event.get("envelope")
    if not isinstance(envelope, dict):
        return None
    data = envelope.get("dataMessage")
    if not isinstance(data, dict):
        return None
    text = data.get("message")
    text = text if isinstance(text, str) else ""
    attachments = _attachments(data.get("attachments"))
    if not text.strip() and not attachments:
        return None
    if not text.strip() and attachments:
        text = "The user sent attachments."
    source = (
        envelope.get("sourceNumber")
        or envelope.get("sourceUuid")
        or envelope.get("source")
    )
    if source is None:
        return None
    group_info = data.get("groupInfo")
    group_id = group_info.get("groupId") if isinstance(group_info, dict) else None
    chat_id = str(group_id or source)
    timestamp = envelope.get("timestamp") or data.get("timestamp")
    message_id = f"{source}:{timestamp}" if timestamp is not None else str(source)
    metadata: dict[str, Any] = {
        "message_id": message_id,
        "timestamp": timestamp,
        "group_id": group_id,
        "sender": str(source),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return InboundMessage(
        channel="signal",
        sender_id=str(source),
        chat_id=chat_id,
        content=text,
        media=[item.filename for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=f"signal:{chat_id}",
    )


def _attachments(raw: Any) -> list[Attachment]:
    if not isinstance(raw, list):
        return []
    result: list[Attachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        media_id = item.get("id") or item.get("attachmentId")
        if not media_id:
            continue
        filename = str(item.get("filename") or "attachment")
        mime_type = str(item.get("contentType") or "")
        result.append(
            Attachment(
                filename=filename,
                mime_type=mime_type,
                kind=_kind_from_mime(mime_type),
                file_id=str(media_id),
                size=int(item.get("size") or 0),
            )
        )
    return result


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
