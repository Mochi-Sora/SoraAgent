"""Channel-agnostic message types.

Transports translate their platform payloads into ``InboundMessage`` and render
``OutboundMessage`` in the other direction.  Nothing here imports a platform SDK
or the agent, so channels and the core loop stay independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

STREAM_DELTA_KIND = "stream_delta"
STREAM_END_KIND = "stream_end"


@dataclass
class InboundMessage:
    """A message received from a channel."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_key_override: str | None = None

    @property
    def session_key(self) -> str:
        """The conversation this message belongs to."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """A message to deliver through a channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)


@dataclass
class TurnReply:
    """One handler reply.

    ``delivered`` means the text was already streamed to the channel and the
    manager must not send it again as a separate message.
    """

    text: str
    delivered: bool = False
