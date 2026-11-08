"""iMessage (BlueBubbles) channel runtime."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Any

from ...files import Attachment
from .._util import Dedupe, split_text
from .._webhook import WebhookServer, path_only, query_params
from ..base import BaseChannel
from ..types import InboundMessage, OutboundMessage
from .config import IMessageConfig
from .transport import IMessageTransport, UrllibIMessageTransport


class IMessageChannel(BaseChannel):
    """An iMessage bridge backed by a BlueBubbles server."""

    name = "imessage"
    display_name = "iMessage"

    def __init__(
        self,
        config: IMessageConfig,
        bus,
        *,
        transport: IMessageTransport | None = None,
        server_factory=None,
    ) -> None:
        super().__init__(config, bus)
        self.config: IMessageConfig = config
        self.transport = transport or UrllibIMessageTransport(
            config.server_url,
            config.password,
            request_timeout=config.request_timeout,
        )
        self._server_factory = server_factory or WebhookServer
        self._server: WebhookServer | None = None
        self._dedupe = Dedupe()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        self._server = self._server_factory(
            self.config.webhook_host,
            self.config.webhook_port,
            self.handle_webhook,
            logger=self.log,
        )
        self._server.start()
        try:
            await self._stopped.wait()
        finally:
            server = self._server
            self._server = None
            if server is not None:
                server.stop()

    async def stop(self) -> None:
        self._running = False
        server = self._server
        if server is not None:
            server.stop()
        loop = self._loop
        stopped = self._stopped
        if loop is not None and stopped is not None:
            loop.call_soon_threadsafe(stopped.set)

    async def send(self, msg: OutboundMessage) -> None:
        for chunk in split_text(msg.content, self.config.max_message_chars):
            self.transport.send_text(msg.chat_id, chunk)

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing:
            return
        try:
            self.transport.send_typing(inbound.chat_id)
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("imessage typing indicator failed: %s", exc)

    # ------------------------------------------------------------------
    # webhook
    # ------------------------------------------------------------------

    def handle_webhook(
        self,
        method: str,
        raw_path: str,
        headers: Mapping[str, str],
        body: bytes,
    ):
        if path_only(raw_path) != self.config.webhook_path:
            return 404, {"Content-Type": "text/plain"}, b"not found"
        if method != "POST":
            return 405, {"Content-Type": "text/plain"}, b"method not allowed"
        if self.config.webhook_secret:
            supplied = headers.get("x-webhook-secret") or query_params(
                raw_path
            ).get("secret")
            if supplied != self.config.webhook_secret:
                return 403, {"Content-Type": "text/plain"}, b"invalid secret"
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return 400, {"Content-Type": "text/plain"}, b"invalid json"
        message = parse_event(payload)
        if message is not None:
            threading.Thread(
                target=self._ingest,
                args=(message,),
                name=f"imessage:{message.sender_id}",
                daemon=True,
            ).start()
        return 200, {"Content-Type": "application/json"}, b'{"status":"ok"}'

    def _ingest(self, message: InboundMessage) -> None:
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate imessage message")
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(
            self.bus.publish_inbound(message), loop
        )
        try:
            future.result(timeout=10)
        except Exception:  # one bad message must not stop the webhook
            self.log.exception("failed to publish imessage message")


def parse_event(payload: Any) -> InboundMessage | None:
    """Translate one BlueBubbles ``new-message`` webhook event."""
    if not isinstance(payload, dict) or payload.get("type") != "new-message":
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    if data.get("isFromMe"):
        return None
    if data.get("associatedMessageGuid") or data.get("associatedMessageType"):
        return None
    text = data.get("text")
    text = text if isinstance(text, str) else ""
    attachments = _attachments(data.get("attachments"))
    if not text.strip() and not attachments:
        return None
    if not text.strip() and attachments:
        text = "The user sent attachments."
    handle = data.get("handle")
    if isinstance(handle, dict):
        sender_id = handle.get("address") or handle.get("id")
    else:
        sender_id = handle
    if not sender_id:
        return None
    chats = data.get("chats")
    chat_guid = data.get("chatGuid")
    if not chat_guid and isinstance(chats, list) and chats:
        first = chats[0]
        if isinstance(first, dict):
            chat_guid = first.get("guid")
    chat_guid = chat_guid or sender_id
    metadata: dict[str, Any] = {
        "message_id": data.get("guid"),
        "chat_guid": str(chat_guid),
        "handle": str(sender_id),
        "date_created": data.get("dateCreated"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return InboundMessage(
        channel="imessage",
        sender_id=str(sender_id),
        chat_id=str(chat_guid),
        content=text,
        media=[item.filename for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=f"imessage:{chat_guid}",
    )


def _attachments(raw: Any) -> list[Attachment]:
    if not isinstance(raw, list):
        return []
    result: list[Attachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        filename = str(
            item.get("transferName")
            or item.get("filename")
            or "attachment"
        )
        mime_type = str(item.get("mimeType") or "")
        guid = item.get("guid") or item.get("transferGuid")
        result.append(
            Attachment(
                filename=filename,
                mime_type=mime_type,
                kind=_kind_from_mime(mime_type),
                file_id=str(guid or ""),
                size=int(item.get("totalBytes") or 0),
                error="" if guid else "attachment has no guid",
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
