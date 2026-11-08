"""Google Chat channel runtime."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Any

from ...files import Attachment
from .._http import HttpError, download
from .._util import Dedupe, split_text
from .._webhook import WebhookServer, path_only
from ..base import BaseChannel
from ..types import InboundMessage, OutboundMessage
from .config import GoogleChatConfig
from .transport import GoogleChatError, GoogleChatTransport, UrllibGoogleChatTransport


class GoogleChatChannel(BaseChannel):
    """A Google Chat app using HTTP events and an incoming webhook.

    Google Chat has no typing-indicator API.  The synchronous webhook response
    is the visible "preparing" message; the final answer is posted through the
    configured incoming webhook.
    """

    name = "googlechat"
    display_name = "Google Chat"

    def __init__(
        self,
        config: GoogleChatConfig,
        bus,
        *,
        transport: GoogleChatTransport | None = None,
        server_factory=None,
    ) -> None:
        super().__init__(config, bus)
        self.config: GoogleChatConfig = config
        self.transport = transport or UrllibGoogleChatTransport(
            config.webhook_url,
            audience=config.audience,
            issuer=config.issuer,
            jwks_url=config.jwks_url,
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
        thread_name = msg.metadata.get("thread_name")
        for chunk in split_text(msg.content, self.config.max_message_chars):
            self.transport.send_webhook(
                chunk,
                thread_name=str(thread_name) if thread_name else None,
            )

    async def send_typing(self, inbound: InboundMessage) -> None:
        # The webhook acknowledgement is sent synchronously in handle_webhook.
        return

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
        if self.config.verify_jwt:
            authorization = headers.get("authorization", "")
            if not authorization.lower().startswith("bearer "):
                return 401, {"Content-Type": "text/plain"}, b"missing bearer token"
            token = authorization.split(" ", 1)[1].strip()
            try:
                self.transport.verify_jwt(token)
            except GoogleChatError as exc:
                self.log.warning("rejected Google Chat event: %s", exc)
                return 401, {"Content-Type": "text/plain"}, b"invalid token"
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return 400, {"Content-Type": "text/plain"}, b"invalid json"
        message = parse_event(payload)
        if message is None:
            return 200, {"Content-Type": "application/json"}, b"{}"
        threading.Thread(
            target=self._ingest,
            args=(message,),
            name=f"googlechat:{message.sender_id}",
            daemon=True,
        ).start()
        response: dict[str, Any] = {}
        if self.config.typing:
            response["text"] = self.config.ack_text
            thread_name = message.metadata.get("thread_name")
            if thread_name:
                response["thread"] = {"name": thread_name}
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps(response, ensure_ascii=False),
        )

    def _ingest(self, message: InboundMessage) -> None:
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate google chat message")
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
        except Exception:  # one bad message must not stop the webhook
            self.log.exception("failed to publish google chat message")

    def _prepare_attachments(self, message: InboundMessage) -> None:
        if not self.config.download_attachments:
            return
        for attachment in message.attachments:
            if attachment.data is not None or attachment.error:
                continue
            if not attachment.file_id.lower().startswith(("http://", "https://")):
                attachment.error = "attachment has no downloadable URL"
                continue
            try:
                attachment.data = download(
                    attachment.file_id,
                    timeout=self.config.request_timeout,
                    max_bytes=self.config.max_attachment_bytes,
                )
                attachment.size = len(attachment.data)
            except HttpError as exc:
                attachment.error = f"download failed: {exc}"


def parse_event(payload: Any) -> InboundMessage | None:
    """Translate one Google Chat ``MESSAGE`` event."""
    if not isinstance(payload, dict) or payload.get("type") != "MESSAGE":
        return None
    message = payload.get("message")
    sender = payload.get("sender")
    space = payload.get("space")
    if not isinstance(message, dict) or not isinstance(sender, dict):
        return None
    if sender.get("type") == "BOT":
        return None
    sender_id = sender.get("name") or sender.get("email")
    if not sender_id:
        return None
    space_name = space.get("name") if isinstance(space, dict) else None
    text = message.get("argumentText") or message.get("text")
    text = text.strip() if isinstance(text, str) else ""
    attachments = _attachments(message.get("attachment"))
    if not text.strip() and not attachments:
        return None
    if not text.strip() and attachments:
        text = "The user sent attachments."
    thread = message.get("thread")
    thread_name = thread.get("name") if isinstance(thread, dict) else None
    metadata: dict[str, Any] = {
        "message_id": message.get("name") or message.get("messageId"),
        "thread_name": thread_name,
        "space_name": space_name,
        "sender_name": sender.get("displayName"),
        "sender_email": sender.get("email"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    chat_id = str(space_name or "googlechat")
    session_key = (
        f"googlechat:{chat_id}:{thread_name}"
        if thread_name
        else f"googlechat:{chat_id}"
    )
    return InboundMessage(
        channel="googlechat",
        sender_id=str(sender_id),
        chat_id=chat_id,
        content=text,
        media=[item.filename for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=session_key,
    )


def _attachments(raw: Any) -> list[Attachment]:
    if not isinstance(raw, list):
        return []
    result: list[Attachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("downloadUri")
        filename = str(item.get("name") or "attachment")
        content_type = str(item.get("contentType") or "")
        result.append(
            Attachment(
                filename=filename,
                mime_type=content_type,
                kind=_kind_from_mime(content_type),
                file_id=str(url or ""),
                error="" if url else "attachment has no download URL",
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
