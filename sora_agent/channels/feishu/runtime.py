"""Feishu / Lark channel runtime."""

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
from .config import FeishuConfig
from .transport import FeishuTransport, UrllibFeishuTransport


class FeishuChannel(BaseChannel):
    """A Feishu/Lark custom app receiving event callbacks."""

    name = "feishu"
    display_name = "Feishu / Lark"

    def __init__(
        self,
        config: FeishuConfig,
        bus,
        *,
        transport: FeishuTransport | None = None,
        server_factory=None,
    ) -> None:
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self.transport = transport or UrllibFeishuTransport(
            config.app_id,
            config.app_secret,
            config.resolved_api_base,
            request_timeout=config.request_timeout,
        )
        self._server_factory = server_factory or WebhookServer
        self._server: WebhookServer | None = None
        self._dedupe = Dedupe()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._reactions: dict[tuple[str, str], str] = {}

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
        try:
            for chunk in split_text(msg.content, self.config.max_message_chars):
                self.transport.send_text(msg.chat_id, chunk)
        finally:
            message_id = msg.metadata.get("message_id")
            if message_id:
                self._clear_reaction(msg.chat_id, str(message_id))

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing:
            return
        message_id = inbound.metadata.get("message_id")
        if not message_id:
            return
        key = (inbound.chat_id, str(message_id))
        if key in self._reactions:
            return
        try:
            reaction_id = self.transport.add_reaction(
                str(message_id),
                self.config.reaction_emoji,
            )
            self._reactions[key] = reaction_id or "none"
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("feishu reaction indicator failed: %s", exc)
            self._reactions[key] = "none"

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
            return 400, {"Content-Type": "text/plain"}, b"invalid event"
        if payload.get("encrypt") is not None:
            return 400, {"Content-Type": "text/plain"}, b"encrypted events are not supported"
        if payload.get("type") == "url_verification":
            if str(payload.get("token") or "") != self.config.verification_token:
                return 403, {"Content-Type": "text/plain"}, b"invalid token"
            return (
                200,
                {"Content-Type": "application/json"},
                json.dumps({"challenge": payload.get("challenge", "")}),
            )
        header = payload.get("header")
        if not isinstance(header, dict):
            return 200, {"Content-Type": "application/json"}, b'{"code":0}'
        if str(header.get("token") or "") != self.config.verification_token:
            return 403, {"Content-Type": "text/plain"}, b"invalid token"
        message = parse_event(payload)
        if message is not None:
            threading.Thread(
                target=self._ingest,
                args=(message,),
                name=f"feishu:{message.sender_id}",
                daemon=True,
            ).start()
        return 200, {"Content-Type": "application/json"}, b'{"code":0}'

    def _ingest(self, message: InboundMessage) -> None:
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate feishu message")
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
            self.log.exception("failed to publish feishu message")

    def _prepare_attachments(self, message: InboundMessage) -> None:
        if not self.config.download_attachments:
            return
        message_id = message.metadata.get("message_id")
        for attachment in message.attachments:
            if attachment.data is not None or attachment.error:
                continue
            if not message_id:
                attachment.error = "message id is unavailable"
                continue
            if attachment.size and attachment.size > self.config.max_attachment_bytes:
                attachment.error = (
                    f"attachment is above the "
                    f"{self.config.max_attachment_bytes} byte limit"
                )
                continue
            resource_type = str(
                attachment.metadata.get("resource_type") or "file"
            )
            try:
                attachment.data = self.transport.download_resource(
                    str(message_id),
                    attachment.file_id,
                    resource_type,
                    self.config.max_attachment_bytes,
                )
                attachment.size = len(attachment.data)
            except Exception as exc:  # noqa: BLE001 - report on the attachment
                attachment.error = f"download failed: {exc}"

    def _clear_reaction(self, chat_id: str, message_id: str) -> None:
        reaction_id = self._reactions.pop((chat_id, message_id), None)
        if not reaction_id or reaction_id == "none":
            return
        try:
            self.transport.delete_reaction(message_id, reaction_id)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            self.log.debug("feishu reaction cleanup failed: %s", exc)


def parse_event(payload: Any) -> InboundMessage | None:
    """Translate one Feishu event-v2 message callback."""
    if not isinstance(payload, dict):
        return None
    header = payload.get("header")
    event = payload.get("event")
    if not isinstance(header, dict) or not isinstance(event, dict):
        return None
    if header.get("event_type") != "im.message.receive_v1":
        return None
    sender = event.get("sender")
    message = event.get("message")
    if not isinstance(sender, dict) or not isinstance(message, dict):
        return None
    if sender.get("sender_type") == "app":
        return None
    sender_id = sender.get("sender_id")
    sender_id = sender_id if isinstance(sender_id, dict) else {}
    open_id = sender_id.get("open_id") or sender_id.get("user_id")
    chat_id = message.get("chat_id")
    if not open_id or not chat_id:
        return None
    content = _decode_content(message.get("content"))
    message_type = str(message.get("message_type") or "")
    text = ""
    attachments: list[Attachment] = []
    if message_type == "text":
        text = str(content.get("text") or "")
    elif message_type == "post":
        text = _post_text(content)
    elif message_type == "image":
        image_key = content.get("image_key")
        if image_key:
            attachments.append(
                Attachment(
                    filename="image",
                    kind="image",
                    file_id=str(image_key),
                    metadata={"resource_type": "image"},
                )
            )
    elif message_type in {"file", "audio", "media"}:
        file_key = content.get("file_key")
        if file_key:
            filename = str(content.get("file_name") or message_type)
            attachments.append(
                Attachment(
                    filename=filename,
                    kind="audio" if message_type == "audio" else "document",
                    file_id=str(file_key),
                    metadata={"resource_type": "file"},
                )
            )
    text = re.sub(r"@_user_\d+", "", text).strip()
    if not text and not attachments:
        return None
    if not text and attachments:
        text = "The user sent attachments."
    metadata: dict[str, Any] = {
        "message_id": message.get("message_id"),
        "chat_id": str(chat_id),
        "chat_type": message.get("chat_type"),
        "message_type": message_type,
        "sender": str(open_id),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return InboundMessage(
        channel="feishu",
        sender_id=str(open_id),
        chat_id=str(chat_id),
        content=text,
        media=[item.file_id for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=f"feishu:{chat_id}",
    )


def _decode_content(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _post_text(content: dict[str, Any]) -> str:
    parts: list[str] = []
    title = content.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    paragraphs = content.get("content")
    if not isinstance(paragraphs, list):
        return "\n".join(parts)
    for paragraph in paragraphs:
        if not isinstance(paragraph, list):
            continue
        line: list[str] = []
        for element in paragraph:
            if not isinstance(element, dict):
                continue
            text = element.get("text") or element.get("content")
            if isinstance(text, str):
                line.append(text)
            href = element.get("href")
            if isinstance(href, str) and href:
                line.append(f"({href})")
        if line:
            parts.append("".join(line))
    return "\n".join(parts)
