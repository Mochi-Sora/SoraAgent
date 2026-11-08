"""WhatsApp Cloud API channel runtime."""

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
from .config import WhatsAppConfig
from .transport import UrllibWhatsAppTransport, WhatsAppTransport


class WhatsAppChannel(BaseChannel):
    """A WhatsApp Cloud API bot receiving signed webhook events."""

    name = "whatsapp"
    display_name = "WhatsApp"

    def __init__(
        self,
        config: WhatsAppConfig,
        bus,
        *,
        transport: WhatsAppTransport | None = None,
        server_factory=None,
    ) -> None:
        super().__init__(config, bus)
        self.config: WhatsAppConfig = config
        self.transport = transport or UrllibWhatsAppTransport(
            config.access_token,
            config.api_base,
            app_secret=config.app_secret,
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
            self.transport.send_text(
                self.config.phone_number_id,
                msg.chat_id,
                chunk,
                reply_to=msg.reply_to,
            )

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing:
            return
        message_id = inbound.metadata.get("message_id")
        if not message_id:
            return
        try:
            self.transport.mark_read(
                self.config.phone_number_id,
                str(message_id),
                typing=True,
            )
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("whatsapp typing indicator failed: %s", exc)

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
        if method == "GET":
            params = query_params(raw_path)
            if (
                params.get("hub.mode") == "subscribe"
                and params.get("hub.verify_token") == self.config.verify_token
            ):
                challenge = params.get("hub.challenge", "")
                return 200, {"Content-Type": "text/plain"}, challenge
            return 403, {"Content-Type": "text/plain"}, b"verification failed"
        if method != "POST":
            return 405, {"Content-Type": "text/plain"}, b"method not allowed"
        if self.config.require_signature:
            signature = headers.get("x-hub-signature-256", "")
            if not signature or not self.transport.verify_signature(body, signature):
                return 403, {"Content-Type": "text/plain"}, b"invalid signature"
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return 400, {"Content-Type": "text/plain"}, b"invalid json"
        try:
            messages = parse_webhook(payload)
        except Exception:
            self.log.exception("failed to parse whatsapp webhook")
            return 200, {"Content-Type": "application/json"}, b'{"status":"ignored"}'
        for message in messages:
            threading.Thread(
                target=self._ingest,
                args=(message,),
                name=f"whatsapp:{message.sender_id}",
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
            self.log.debug("ignored duplicate whatsapp message")
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
        except Exception:
            self.log.exception("failed to publish whatsapp message")

    def _prepare_attachments(self, message: InboundMessage) -> None:
        if not self.config.download_attachments:
            return
        for attachment in message.attachments:
            if attachment.data is not None or attachment.error:
                continue
            try:
                data, mime_type = self.transport.download_media(
                    attachment.file_id,
                    self.config.max_attachment_bytes,
                )
                attachment.data = data
                attachment.size = len(data)
                if not attachment.mime_type:
                    attachment.mime_type = mime_type
            except Exception as exc:  # noqa: BLE001 - report on the attachment
                attachment.error = f"download failed: {exc}"


def parse_webhook(payload: Any) -> list[InboundMessage]:
    """Translate a WhatsApp webhook body into inbound messages."""
    if not isinstance(payload, dict):
        return []
    if payload.get("object") != "whatsapp_business_account":
        return []
    result: list[InboundMessage] = []
    entries = payload.get("entry")
    if not isinstance(entries, list):
        return []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes")
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not isinstance(change, dict):
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            raw_metadata = value.get("metadata")
            metadata: dict[str, Any] = (
                raw_metadata if isinstance(raw_metadata, dict) else {}
            )
            phone_number_id = metadata.get("phone_number_id")
            messages = value.get("messages")
            if not isinstance(messages, list):
                continue
            for message in messages:
                parsed = parse_message(message, phone_number_id)
                if parsed is not None:
                    result.append(parsed)
    return result


def parse_message(message: Any, phone_number_id: Any = None) -> InboundMessage | None:
    """Translate one WhatsApp ``messages[]`` entry."""
    if not isinstance(message, dict):
        return None
    sender = message.get("from")
    if sender is None:
        return None
    message_id = message.get("id")
    kind = str(message.get("type") or "")
    text = ""
    attachments: list[Attachment] = []
    if kind == "text":
        body = message.get("text")
        if isinstance(body, dict):
            text = str(body.get("body") or "")
    elif kind in {"image", "video", "document", "audio", "voice", "sticker"}:
        media = message.get(kind)
        if isinstance(media, dict):
            text = str(media.get("caption") or "")
            mime_type = str(media.get("mime_type") or "")
            filename = str(
                media.get("filename")
                or _default_filename(kind, mime_type)
            )
            media_id = media.get("id")
            if media_id:
                attachments.append(
                    Attachment(
                        filename=filename,
                        mime_type=mime_type,
                        kind=_kind_for(kind, mime_type),
                        file_id=str(media_id),
                    )
                )
    elif kind in {"button", "interactive"}:
        button = message.get("button")
        if isinstance(button, dict):
            text = str(button.get("text") or "")
        interactive = message.get("interactive")
        if isinstance(interactive, dict):
            reply = interactive.get("button_reply") or interactive.get("list_reply")
            if isinstance(reply, dict):
                text = str(reply.get("title") or reply.get("id") or "")
    elif kind == "location":
        location = message.get("location")
        if isinstance(location, dict):
            text = (
                f"Location: {location.get('latitude')}, "
                f"{location.get('longitude')}"
            )
    if not text.strip() and not attachments:
        return None
    if not text.strip() and attachments:
        text = "The user sent attachments."
    metadata: dict[str, Any] = {
        "message_id": message_id,
        "type": kind,
        "timestamp": message.get("timestamp"),
        "phone_number_id": phone_number_id,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return InboundMessage(
        channel="whatsapp",
        sender_id=str(sender),
        chat_id=str(sender),
        content=text,
        media=[item.file_id for item in attachments],
        attachments=attachments,
        metadata=metadata,
    )


def _default_filename(kind: str, mime_type: str) -> str:
    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "application/pdf": ".pdf",
    }
    return f"{kind}{suffixes.get(mime_type.lower(), '')}"


def _kind_for(kind: str, mime_type: str) -> str:
    if kind == "sticker":
        return "image"
    if kind in {"audio", "voice"}:
        return "audio"
    if kind == "video":
        return "video"
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
