"""Slack channel runtime: Socket Mode receive loop and Web API sends."""

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
from .config import SlackConfig
from .transport import SlackError, SlackTransport, UrllibSlackTransport

_IGNORED_SUBTYPES = {
    "bot_message",
    "message_changed",
    "message_deleted",
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "channel_archive",
    "channel_unarchive",
    "group_join",
    "group_leave",
    "pinned_item",
    "unpinned_item",
}


class SlackChannel(BaseChannel):
    """A Slack app using Socket Mode for inbound events."""

    name = "slack"
    display_name = "Slack"

    def __init__(
        self,
        config: SlackConfig,
        bus,
        *,
        transport: SlackTransport | None = None,
        ws_factory=None,
        sleeper=time.sleep,
    ) -> None:
        super().__init__(config, bus)
        self.config: SlackConfig = config
        self.transport = transport or UrllibSlackTransport(
            config.bot_token,
            config.app_token,
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
        self._bot_user_id: str | None = None
        self._typing: dict[tuple[str, str], tuple[str, str | None]] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        try:
            identity = self.transport.auth_test()
            self._bot_user_id = str(identity.get("user_id") or "") or None
        except Exception as exc:  # noqa: BLE001 - auth.test is diagnostic
            self.log.debug("slack auth.test failed: %s", exc)
        self._thread = threading.Thread(
            target=self._gateway_loop,
            name="slack-socket-mode",
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
        thread_ts = _thread_ts(msg.metadata)
        try:
            for chunk in split_text(msg.content, self.config.max_message_chars):
                self.transport.post_message(
                    msg.chat_id,
                    chunk,
                    thread_ts=thread_ts,
                )
        finally:
            self._clear_typing(msg.chat_id, thread_ts or "")

    async def send_typing(self, inbound: InboundMessage) -> None:
        if not self.config.typing or self.config.typing_mode == "off":
            return
        thread_ts = _thread_ts(inbound.metadata) or ""
        key = (inbound.chat_id, thread_ts)
        if key in self._typing:
            return
        if self.config.typing_mode in {"auto", "status"} and thread_ts:
            try:
                self.transport.set_status(
                    inbound.chat_id,
                    thread_ts,
                    self.config.ack_text,
                )
                self._typing[key] = ("status", None)
                return
            except Exception as exc:  # noqa: BLE001 - fall back to a message
                self.log.debug("slack status indicator failed: %s", exc)
                if self.config.typing_mode == "status":
                    self._typing[key] = ("none", None)
                    return
        try:
            ts = self.transport.post_message(
                inbound.chat_id,
                self.config.ack_text,
                thread_ts=thread_ts or None,
            )
            self._typing[key] = ("message", ts)
        except Exception as exc:  # noqa: BLE001 - typing is best effort
            self.log.debug("slack preparing message failed: %s", exc)
            self._typing[key] = ("none", None)

    # ------------------------------------------------------------------
    # Socket Mode thread
    # ------------------------------------------------------------------

    def _gateway_loop(self) -> None:
        attempts = 0
        try:
            while self._running:
                ws = None
                try:
                    ws = self._ws_factory(self.transport.open_socket())
                    self._ws = ws
                    attempts = 0
                    self._run_socket(ws)
                except WebSocketClosed as exc:
                    if not self._running:
                        break
                    self.log.warning("slack socket closed: %s", exc)
                except (WebSocketError, OSError, TimeoutError, ValueError) as exc:
                    if not self._running:
                        break
                    self.log.warning("slack socket error: %s", exc)
                except SlackError as exc:
                    if not self._running:
                        break
                    self.log.warning("slack connection error: %s", exc)
                except Exception:  # keep the gateway alive
                    if not self._running:
                        break
                    self.log.exception("slack socket failed")
                finally:
                    self._ws = None
                    if ws is not None:
                        try:
                            ws.close()
                        except Exception as exc:  # noqa: BLE001 - best effort
                            self.log.debug("slack socket close failed: %s", exc)
                attempts += 1
                limit = self.config.reconnect_attempts
                if limit and attempts >= limit:
                    self.log.error("slack reconnect limit reached (%d)", limit)
                    break
                self._sleeper(self.config.reconnect_delay * min(attempts, 5))
        finally:
            self._running = False
            loop = self._loop
            stopped = self._stopped
            if loop is not None and stopped is not None:
                loop.call_soon_threadsafe(stopped.set)

    def _run_socket(self, ws) -> None:
        while self._running:
            payload = ws.recv_json(timeout=None)
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind == "hello":
                continue
            if kind == "disconnect":
                self.log.info("slack requested a reconnect")
                return
            if kind != "events_api":
                continue
            envelope_id = payload.get("envelope_id")
            if envelope_id:
                ws.send_json({"envelope_id": str(envelope_id)})
            event = payload.get("payload", {}).get("event")
            if isinstance(event, dict):
                self._dispatch(event)

    def _dispatch(self, event: dict[str, Any]) -> None:
        message = parse_event(event, self._bot_user_id)
        if message is None:
            return
        if not self.is_allowed(message.sender_id):
            self.log.debug(
                "ignored message from unauthorized sender %s", message.sender_id
            )
            return
        if self._dedupe.seen(message.metadata.get("message_id")):
            self.log.debug("ignored duplicate slack message")
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
        except Exception:  # a full queue must not kill the socket
            self.log.exception("failed to publish slack message")

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

    def _clear_typing(self, channel: str, thread_ts: str) -> None:
        value = self._typing.pop((channel, thread_ts), None)
        if value is None:
            return
        kind, handle = value
        try:
            if kind == "status" and thread_ts:
                self.transport.set_status(channel, thread_ts, "")
            elif kind == "message" and handle:
                self.transport.delete_message(channel, handle)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            self.log.debug("slack typing cleanup failed: %s", exc)

    def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception as exc:  # noqa: BLE001 - best effort
                self.log.debug("slack socket close failed: %s", exc)


def parse_event(
    event: dict[str, Any],
    bot_user_id: str | None = None,
) -> InboundMessage | None:
    """Translate one Slack ``message`` event."""
    if event.get("type") != "message":
        return None
    subtype = event.get("subtype")
    if subtype and subtype != "file_share":
        return None
    if event.get("bot_id"):
        return None
    user = event.get("user")
    channel = event.get("channel")
    if not user or not channel:
        return None
    if bot_user_id and str(user) == str(bot_user_id):
        return None
    text = event.get("text")
    text = text if isinstance(text, str) else ""
    if bot_user_id:
        text = re.sub(rf"<@{re.escape(str(bot_user_id))}>", "", text).strip()
    attachments = _attachments(event.get("files"))
    if not text.strip() and not attachments:
        return None
    if not text.strip() and attachments:
        text = "The user sent attachments."
    ts = event.get("ts")
    thread_ts = event.get("thread_ts")
    metadata: dict[str, Any] = {
        "message_id": ts,
        "channel": channel,
        "thread_ts": thread_ts,
        "user": user,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    session_key = (
        f"slack:{channel}:{thread_ts}" if thread_ts else f"slack:{channel}"
    )
    return InboundMessage(
        channel="slack",
        sender_id=str(user),
        chat_id=str(channel),
        content=text,
        media=[item.file_id for item in attachments],
        attachments=attachments,
        metadata=metadata,
        session_key_override=session_key,
    )


def _thread_ts(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("thread_ts")
    return None if value is None else str(value)


def _attachments(raw: Any) -> list[Attachment]:
    if not isinstance(raw, list):
        return []
    result: list[Attachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url_private_download") or item.get("url_private")
        if not url:
            continue
        if item.get("mode") == "tombstone":
            continue
        filename = str(item.get("name") or item.get("title") or "attachment")
        mime_type = str(item.get("mimetype") or "")
        result.append(
            Attachment(
                filename=filename,
                mime_type=mime_type,
                kind=_kind_from_mime(mime_type),
                file_id=str(url),
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
