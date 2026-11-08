"""Microsoft Teams channel runtime."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Mapping
from typing import Any

from ...files import Attachment
from .._util import Dedupe, split_text
from .._webhook import WebhookServer, path_only
from ..base import BaseChannel
from ..types import InboundMessage, OutboundMessage
from .config import TeamsConfig
from .transport import TeamsError, TeamsTransport, UrllibTeamsTransport


class TeamsChannel(BaseChannel):
    """A Microsoft Teams bot receiving Bot Framework activities."""

    name = "teams"
    display_name = "Microsoft Teams"

    def __init__(
        self,
        config: TeamsConfig,
        bus,
        *,
        transport: TeamsTransport | None = None,
        server_factory=None,
    ) -> None:
        super().__init__(config, bus)
        self.config: TeamsConfig = config
        self.transport = transport or UrllibTeamsTransport(
            config.app_id,
            config.app_password,
            token_url=config.resolved_token_url,
            openid_config_url=config.openid_config_url,
            scope=config.scope,
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
        service_url = msg.metadata.get("service_url")
        conversation_id = msg.metadata.get("conversation_id") or msg.chat_id
        if not service_url:
            raise TeamsError("teams outbound message has no service_url")
        for chunk in split_text(msg.content, self.config.max_message_chars):
            self.transport.send_text(
                str(service_url),
                str(conversation_id),
                chunk,
                reply_to=msg.reply_to,
            )

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing:
            return
        service_url = inbound.metadata.get("service_url")
        conversation_id = inbound.metadata.get("conversation_id") or inbound.chat_id
        if not service_url:
            return
        try:
            self.transport.send_typing(
                str(service_url),
                str(conversation_id),
            )
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("teams typing indicator failed: %s", exc)

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
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return 400, {"Content-Type": "text/plain"}, b"invalid json"
        if not isinstance(payload, dict):
            return 400, {"Content-Type": "text/plain"}, b"invalid activity"
        if self.config.verify_jwt:
            authorization = headers.get("authorization", "")
            if not authorization.lower().startswith("bearer "):
                return 401, {"Content-Type": "text/plain"}, b"missing bearer token"
            token = authorization.split(" ", 1)[1].strip()
            service_url = payload.get("serviceUrl")
            try:
                self.transport.verify_jwt(
                    token,
                    str(service_url) if service_url else None,
                )
            except TeamsError as exc:
                self.log.warning("rejected Teams activity: %s", exc)
                return 401, {"Content-Type": "text/plain"}, b"invalid token"
        message = parse_activity(payload, self.config.app_id)
        if message is None:
            return 200, {"Content-Type": "application/json"}, b'{"status":"ignored"}'
        threading.Thread(
            target=self._ingest,
            args=(message,),
            name=f"teams:{message.sender_id}",
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
            self.log.debug("ignored duplicate teams message")
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
            self.log.exception("failed to publish teams message")

    def _prepare_attachments(self, message: InboundMessage) -> None:
        if not self.config.download_attachments:
            return
        for attachment in message.attachments:
            if attachment.data is not None or attachment.error:
                continue
            if not attachment.file_id.lower().startswith(("http://", "https://")):
                attachment.error = "attachment has no downloadable URL"
                continue
            if attachment.size and attachment.size > self.config.max_attachment_bytes:
                attachment.error = (
                    f"attachment is above the "
                    f"{self.config.max_attachment_bytes} byte limit"
                )
                continue
            try:
                attachment.data = self.transport.download(
                    attachment.file_id,
                    self.config.max_attachment_bytes,
                )
                attachment.size = len(attachment.data)
            except Exception as exc:  # noqa: BLE001 - report on the attachment
                attachment.error = f"download failed: {exc}"


def parse_activity(activity: Any, app_id: str = "") -> InboundMessage | None:
    """Translate one Bot Framework message activity."""
    if not isinstance(activity, dict):
        return None
    if activity.get("type") != "message":
        return None
    sender = activity.get("from")
    conversation = activity.get("conversation")
    if not isinstance(sender, dict) or not isinstance(conversation, dict):
        return None
    if sender.get("role") == "bot":
        return None
    sender_id = sender.get("id")
    conversation_id = conversation.get("id")
    if sender_id is None or conversation_id is None:
        return None
    if app_id and str(sender_id) == str(app_id):
        return None
    text = activity.get("text")
    text = text if isinstance(text, str) else ""
    text = re.sub(r"<at>.*?</at>", "", text, flags=re.IGNORECASE).strip()
    attachments = _attachments(activity.get("attachments"))
    if not text and not attachments:
        return None
    if not text and attachments:
        text = "The user sent attachments."
    service_url = activity.get("serviceUrl")
    metadata: dict[str, Any] = {
        "message_id": activity.get("id"),
        "service_url": service_url,
        "conversation_id": conversation_id,
        "channel_id": activity.get("channelId"),
        "reply_to_id": activity.get("replyToId"),
        "sender_name": sender.get("name"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return InboundMessage(
        channel="teams",
        sender_id=str(sender_id),
        chat_id=str(conversation_id),
        content=text,
        media=[item.file_id for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=f"teams:{conversation_id}",
    )


def _attachments(raw: Any) -> list[Attachment]:
    if not isinstance(raw, list):
        return []
    result: list[Attachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content_type = str(item.get("contentType") or "")
        if content_type.lower().startswith("text/html"):
            # Teams repeats the message text in an HTML attachment.
            continue
        url = item.get("contentUrl")
        if not url:
            continue
        filename = str(item.get("name") or "attachment")
        result.append(
            Attachment(
                filename=filename,
                mime_type=content_type,
                kind=_kind_from_mime(content_type),
                file_id=str(url),
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
