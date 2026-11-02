"""Chat-driven human approval for consequential tool calls.

``ApprovalGate.request`` is synchronous and blocking, which is correct for a CLI
but wrong for a chat: the agent runs in a worker thread while the answer arrives
asynchronously on the bus.  The broker bridges the two — it publishes the prompt,
the manager intercepts the reply, and the worker thread wakes with a decision.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..approval import ApprovalGate, AuditEntry
from .bus import MessageBus
from .types import InboundMessage, OutboundMessage

APPROVAL_KIND = "approval"
APPROVAL_CALLBACK_KIND = "approval_callback"
_PROMPT_PUBLISH_TIMEOUT = 10.0

_YES = frozenset({"y", "yes", "yeah", "approve", "approved", "ok", "okay", "go"})
_NO = frozenset({"n", "no", "nope", "reject", "rejected", "deny", "denied", "stop"})


def parse_decision(text: str) -> bool | None:
    """Return ``True``/``False`` for a clear yes/no reply, ``None`` otherwise."""
    cleaned = str(text or "").strip().lower().strip(".!?")
    if not cleaned:
        return None
    word = cleaned.split()[0]
    if word in _YES:
        return True
    if word in _NO:
        return False
    return None


def render_request(action: str, detail: str) -> str:
    return (
        "Approval required.\n"
        f"Action: {action}\n"
        f"Detail: {detail}\n"
        "Reply yes to approve or no to reject."
    )


@dataclass
class PendingApproval:
    """One request waiting for a human reply."""

    session_key: str
    action: str
    detail: str
    channel: str
    chat_id: str
    actor: str
    reply_to: str | None = None
    approval_id: str = ""
    event: threading.Event = field(default_factory=threading.Event)
    decision: bool | None = None


class ApprovalBroker:
    """Owns pending approvals and the loop that publishes their prompts."""

    def __init__(
        self,
        bus: MessageBus,
        *,
        timeout: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("approval timeout must be > 0")
        self.bus = bus
        self.timeout = float(timeout)
        self.log = logger or logging.getLogger("sora_agent.channels.approval")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the gateway loop; called by ``ChannelManager.serve``."""
        self._loop = loop

    def consume(self, inbound: InboundMessage) -> bool:
        """Resolve a pending approval from a reply.  True means it was consumed.

        A message that is not a clear yes/no is left for the agent, so a user can
        ask a follow-up question while an approval is pending.
        """
        with self._lock:
            pending = self._pending.get(inbound.session_key)
        if pending is None:
            return False
        supplied = str(inbound.metadata.get("approval_id") or "")
        if supplied and supplied != pending.approval_id:
            # A button from an older prompt must not decide this request.
            return False
        decision = parse_decision(inbound.content)
        if decision is None:
            return False
        with self._lock:
            if self._pending.get(inbound.session_key) is not pending:
                return False
            pending.decision = decision
            del self._pending[inbound.session_key]
        pending.event.set()
        return True

    def cancel_all(self) -> None:
        """Deny everything still waiting, so shutdown never hangs on a thread."""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.decision = False
            item.event.set()

    def gate(self, inbound: InboundMessage) -> ChannelApprovalGate:
        """Build the gate for one conversation turn."""
        message_id = inbound.metadata.get("message_id")
        return ChannelApprovalGate(
            self,
            session_key=inbound.session_key,
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            actor=inbound.sender_id,
            reply_to=None if message_id is None else str(message_id),
        )

    def request(self, pending: PendingApproval) -> bool:
        """Publish the prompt and block until a decision or timeout."""
        if not pending.approval_id:
            pending.approval_id = uuid.uuid4().hex[:12]
        with self._lock:
            if pending.session_key in self._pending:
                self.log.warning(
                    "approval already pending for %s", pending.session_key
                )
                return False
            self._pending[pending.session_key] = pending
        try:
            self._publish(pending)
            decided = pending.event.wait(self.timeout)
        finally:
            with self._lock:
                if self._pending.get(pending.session_key) is pending:
                    del self._pending[pending.session_key]
        if not decided:
            self.log.warning("approval for %s timed out", pending.action)
            return False
        return bool(pending.decision)

    def _publish(self, pending: PendingApproval) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("approval broker is not bound to a running loop")
        message = OutboundMessage(
            channel=pending.channel,
            chat_id=pending.chat_id,
            content=render_request(pending.action, pending.detail),
            reply_to=pending.reply_to,
            metadata={
                "kind": APPROVAL_KIND,
                "action": pending.action,
                "approval_id": pending.approval_id,
            },
            buttons=[["Approve", "Reject"]],
        )
        # The caller is a worker thread, so hop back onto the event loop.
        asyncio.run_coroutine_threadsafe(
            self.bus.publish_outbound(message), loop
        ).result(timeout=_PROMPT_PUBLISH_TIMEOUT)


class ChannelApprovalGate(ApprovalGate):
    """An ``ApprovalGate`` whose question goes to the chat, not stdin."""

    def __init__(
        self,
        broker: ApprovalBroker,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        actor: str,
        reply_to: str | None = None,
    ) -> None:
        super().__init__()
        self.broker = broker
        self.session_key = session_key
        self.channel = channel
        self.chat_id = chat_id
        self.actor = actor
        self.reply_to = reply_to

    def request(self, action: str, detail: str) -> bool:
        approved = self.broker.request(
            PendingApproval(
                session_key=self.session_key,
                action=action,
                detail=detail,
                channel=self.channel,
                chat_id=self.chat_id,
                actor=self.actor,
                reply_to=self.reply_to,
            )
        )
        self.audit.append(
            AuditEntry(
                time.time(),
                action,
                self.actor,
                "approved" if approved else "rejected",
                detail,
            )
        )
        return approved
