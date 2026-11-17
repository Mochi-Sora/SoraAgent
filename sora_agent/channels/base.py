"""Base channel interface for chat platforms."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..files import Attachment
from .bus import MessageBus
from .types import InboundMessage, OutboundMessage


class BaseChannel(ABC):
    """One transport.  ``start`` blocks until the channel stops.

    Subclasses handle platform specifics and call :meth:`_handle_message` for
    every accepted inbound message.  Authorization lives here so every channel
    enforces the same ``allow_from`` rule.
    """

    name: str = "base"
    display_name: str = "Base"
    max_attachments: int = 20
    max_total_attachment_bytes: int = 100 * 1024 * 1024
    #: Whether the transport can edit a live message as text arrives.
    supports_streaming: bool = False

    def __init__(self, config: Any, bus: MessageBus) -> None:
        self.config = config
        self.bus = bus
        self.log = logging.getLogger(f"sora_agent.channels.{self.name}")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """Start listening and block until :meth:`stop` is called."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and release resources.  Must be idempotent."""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Deliver one message, raising if the platform rejects it."""

    async def send_typing(self, inbound: InboundMessage) -> None:
        """Best-effort typing indicator.  Channels may ignore this."""
        return

    async def acknowledge_callback(
        self, inbound: InboundMessage, consumed: bool
    ) -> None:
        """Best-effort acknowledgement for a platform callback/button press.

        ``consumed`` is true only when the manager accepted the callback as an
        approval decision.  Channels may ignore this hook.
        """
        return

    def is_allowed(self, sender_id: str) -> bool:
        """Check ``config.allow_from``.

        ``"*"`` allows everyone and an empty list denies everyone, matching the
        conservative default: a token alone is not permission to talk to it.
        """
        allow = getattr(self.config, "allow_from", ())
        if isinstance(allow, str):
            allow = allow.split(",")
        allowed = {str(item).strip() for item in allow if str(item).strip()}
        return "*" in allowed or str(sender_id) in allowed

    def limit_attachments(self, attachments: list[Attachment]) -> list[Attachment]:
        """Apply count and known-total-byte caps before any download.

        Unknown sizes are still allowed up to the per-file limit enforced by
        each channel; this cap prevents a payload with hundreds of URLs from
        starting hundreds of downloads.
        """
        selected: list[Attachment] = []
        total = 0
        for attachment in attachments:
            if len(selected) >= max(1, int(self.max_attachments)):
                attachment.error = (
                    f"attachment count limit reached ({self.max_attachments})"
                )
                continue
            size = max(0, int(attachment.size or 0))
            if (
                size
                and total + size > max(1, int(self.max_total_attachment_bytes))
            ):
                attachment.error = (
                    "attachment total-byte limit reached "
                    f"({self.max_total_attachment_bytes} bytes)"
                )
                continue
            total += size
            selected.append(attachment)
        return selected

    def enforce_attachment_totals(self, message: Any) -> None:
        """Apply the total-byte cap after unknown-size downloads complete."""
        total = 0
        kept: list[Attachment] = []
        for attachment in message.attachments:
            size = (
                len(attachment.data)
                if attachment.data is not None
                else max(0, int(attachment.size or 0))
            )
            if (
                size
                and total + size > max(1, int(self.max_total_attachment_bytes))
            ):
                attachment.data = None
                attachment.error = (
                    "attachment total-byte limit reached "
                    f"({self.max_total_attachment_bytes} bytes)"
                )
                kept.append(attachment)
                continue
            total += size
            kept.append(attachment)
        message.attachments = kept

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        attachments: list[Attachment] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> bool:
        """Validate and publish an inbound message.  Returns whether it passed."""
        if not self.is_allowed(sender_id):
            self.log.debug("ignored message from unauthorized sender %s", sender_id)
            return False
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender_id=str(sender_id),
                chat_id=str(chat_id),
                content=content,
                media=list(media or []),
                attachments=list(attachments or []),
                metadata=dict(metadata or {}),
                session_key_override=session_key,
            )
        )
        return True
