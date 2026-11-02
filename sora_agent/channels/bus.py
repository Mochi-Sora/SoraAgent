"""In-process queues that decouple transports from the agent."""

from __future__ import annotations

import asyncio

from .types import InboundMessage, OutboundMessage


class MessageBus:
    """Two queues: what came in from a channel, and what must go back out.

    The gateway runs one dispatcher over ``inbound`` and one deliverer over
    ``outbound``, so a slow platform never blocks the model loop.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize)

    async def publish_inbound(self, message: InboundMessage) -> None:
        await self.inbound.put(message)

    async def publish_outbound(self, message: OutboundMessage) -> None:
        await self.outbound.put(message)

    async def next_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def next_outbound(self) -> OutboundMessage:
        return await self.outbound.get()

    def pending(self) -> tuple[int, int]:
        """Return ``(inbound, outbound)`` queue depths, for health checks."""
        return self.inbound.qsize(), self.outbound.qsize()
