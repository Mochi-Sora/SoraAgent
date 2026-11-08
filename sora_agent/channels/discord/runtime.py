"""Discord channel runtime: gateway receive loop, REST sends, typing."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any

from ...files import Attachment
from .._util import Dedupe, split_text
from .._ws import WebSocketClient, WebSocketClosed, WebSocketError
from ..base import BaseChannel
from ..types import InboundMessage, OutboundMessage
from .config import DiscordConfig
from .transport import DiscordTransport, UrllibDiscordTransport


class DiscordChannel(BaseChannel):
    """A Discord bot using the gateway WebSocket and REST API."""

    name = "discord"
    display_name = "Discord"

    def __init__(
        self,
        config: DiscordConfig,
        bus,
        *,
        transport: DiscordTransport | None = None,
        ws_factory=None,
        sleeper=time.sleep,
    ) -> None:
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self.transport = transport or UrllibDiscordTransport(
            config.token,
            config.api_base,
            request_timeout=config.request_timeout,
        )
        self._ws_factory = ws_factory or (
            lambda url: WebSocketClient.connect(
                url,
                timeout=config.request_timeout,
                log=self.log,
            )
        )
        self._sleeper = sleeper
        self._dedupe = Dedupe()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ws = None
        self._seq: int | None = None
        self._bot_id: str | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        self._thread = threading.Thread(
            target=self._gateway_loop,
            name="discord-gateway",
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
        self._close_ws()
        loop = self._loop
        stopped = self._stopped
        if loop is not None and stopped is not None:
            loop.call_soon_threadsafe(stopped.set)

    async def send(self, msg: OutboundMessage) -> None:
        chunks = split_text(msg.content, self.config.max_message_chars)
        for chunk in chunks:
            self.transport.send_message(
                msg.chat_id,
                chunk,
                reply_to=msg.reply_to,
            )

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing:
            return
        try:
            self.transport.send_typing(inbound.chat_id)
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("discord typing indicator failed: %s", exc)

    # ------------------------------------------------------------------
    # gateway thread
    # ------------------------------------------------------------------

    def _gateway_loop(self) -> None:
        attempts = 0
        try:
            while self._running:
                ws = None
                try:
                    base = self.config.gateway_url or self.transport.gateway_url()
                    ws = self._ws_factory(_gateway_url(base))
                    self._ws = ws
                    attempts = 0
                    self._run_socket(ws)
                except WebSocketClosed as exc:
                    if exc.code == 4004:
                        self.log.error("discord rejected the bot token")
                        break
                    if not self._running:
                        break
                    self.log.warning("discord gateway closed: %s", exc)
                except (WebSocketError, OSError, TimeoutError, ValueError) as exc:
                    if not self._running:
                        break
                    self.log.warning("discord gateway error: %s", exc)
                except Exception:  # keep the gateway alive
                    if not self._running:
                        break
                    self.log.exception("discord gateway failed")
                finally:
                    self._ws = None
                    if ws is not None:
                        try:
                            ws.close()
                        except Exception as exc:  # noqa: BLE001 - best effort
                            self.log.debug("discord socket close failed: %s", exc)
                attempts += 1
                limit = self.config.reconnect_attempts
                if limit and attempts >= limit:
                    self.log.error(
                        "discord gateway reconnect limit reached (%d)", limit
                    )
                    break
                self._sleeper(self.config.reconnect_delay * min(attempts, 5))
        finally:
            self._running = False
            loop = self._loop
            stopped = self._stopped
            if loop is not None and stopped is not None:
                loop.call_soon_threadsafe(stopped.set)

    def _run_socket(self, ws) -> None:
        hello = ws.recv_json(timeout=self.config.request_timeout)
        if not isinstance(hello, dict) or hello.get("op") != 10:
            raise WebSocketError("discord gateway did not send Hello")
        data = hello.get("d") or {}
        interval = float(data.get("heartbeat_interval", 41250)) / 1000.0
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(ws, max(1.0, interval), stop_heartbeat),
            name="discord-heartbeat",
            daemon=True,
        )
        heartbeat.start()
        try:
            ws.send_json(
                {
                    "op": 2,
                    "d": {
                        "token": self.config.token,
                        "intents": self.config.intents,
                        "properties": {
                            "os": "linux",
                            "browser": "sora-agent",
                            "device": "sora-agent",
                        },
                    },
                }
            )
            while self._running:
                try:
                    payload = ws.recv_json(timeout=None)
                except TimeoutError:
                    continue
                if not isinstance(payload, dict):
                    continue
                op = payload.get("op")
                sequence = payload.get("s")
                if sequence is not None:
                    self._seq = int(sequence)
                if op == 1:
                    ws.send_json({"op": 1, "d": self._seq})
                elif op == 7:
                    self.log.info("discord requested a reconnect")
                    return
                elif op == 9:
                    self.log.warning("discord invalidated the session")
                    self._seq = None
                    return
                elif op == 0:
                    self._dispatch(payload.get("t"), payload.get("d"))
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=2)

    def _heartbeat(self, ws, interval: float, stop: threading.Event) -> None:
        while not stop.wait(interval):
            if not self._running:
                return
            try:
                ws.send_json({"op": 1, "d": self._seq})
            except Exception:  # noqa: BLE001 - the reader will notice
                return

    def _dispatch(self, event: Any, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if event == "READY":
            user = data.get("user")
            if isinstance(user, dict):
                self._bot_id = str(user.get("id") or "") or None
            return
        if event != "MESSAGE_CREATE":
            return
        message = parse_message(data, self._bot_id)
        if message is None:
            return
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate discord message")
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
            self.log.exception("failed to publish discord message")

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
                attachment.data = self.transport.download(
                    attachment.file_id,
                    self.config.max_attachment_bytes,
                )
                attachment.size = len(attachment.data)
            except Exception as exc:  # noqa: BLE001 - report on the attachment
                attachment.error = f"download failed: {exc}"

    def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception as exc:  # noqa: BLE001 - best effort
                self.log.debug("discord socket close failed: %s", exc)


def _gateway_url(base: str) -> str:
    if "?" in base:
        return f"{base}&v=10&encoding=json"
    return f"{base}?v=10&encoding=json"


def parse_message(data: Any, bot_id: str | None = None) -> InboundMessage | None:
    """Translate one Discord ``MESSAGE_CREATE`` payload."""
    if not isinstance(data, dict):
        return None
    author = data.get("author")
    if not isinstance(author, dict) or author.get("bot"):
        return None
    sender_id = author.get("id")
    channel_id = data.get("channel_id")
    if sender_id is None or channel_id is None:
        return None
    content = data.get("content")
    content = content if isinstance(content, str) else ""
    if bot_id:
        content = re.sub(rf"<@!?{re.escape(str(bot_id))}>", "", content).strip()
    attachments = _attachments(data.get("attachments"))
    if not content.strip() and not attachments:
        return None
    if not content.strip() and attachments:
        content = "The user sent attachments."
    metadata: dict[str, Any] = {
        "message_id": data.get("id"),
        "channel_id": str(channel_id),
        "guild_id": data.get("guild_id"),
        "author_name": author.get("username"),
    }
    return InboundMessage(
        channel="discord",
        sender_id=str(sender_id),
        chat_id=str(channel_id),
        content=content,
        media=[item.file_id for item in attachments],
        attachments=attachments,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _attachments(raw: Any) -> list[Attachment]:
    if not isinstance(raw, list):
        return []
    result: list[Attachment] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        filename = str(item.get("filename") or "attachment")
        mime_type = str(item.get("content_type") or "")
        result.append(
            Attachment(
                filename=filename,
                mime_type=mime_type,
                kind=_kind_from_mime(mime_type),
                file_id=str(item["url"]),
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
