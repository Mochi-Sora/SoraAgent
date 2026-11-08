"""Tests for chat-driven approvals."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from sora_agent.approval import ApprovalGate
from sora_agent.channels.approval import (
    ApprovalBroker,
    PendingApproval,
    parse_decision,
    render_request,
)
from sora_agent.channels.bus import MessageBus
from sora_agent.channels.types import InboundMessage


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        (" YES! ", True),
        ("approve", True),
        ("ok", True),
        ("no", False),
        ("Reject.", False),
        ("stop", False),
        ("what about PO-2?", None),
        ("", None),
    ],
)
def test_parse_decision(text, expected):
    assert parse_decision(text) is expected


async def _join(worker: threading.Thread, timeout: float = 2.0) -> None:
    """Wait for a worker without blocking the loop it is calling into."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while worker.is_alive() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    worker.join(0)


def _pending(**overrides) -> PendingApproval:
    values: dict[str, Any] = {
        "session_key": "telegram:42",
        "action": "cancel_order",
        "detail": 'ACTION: cancel_order(order_id="PO-1")',
        "channel": "telegram",
        "chat_id": "42",
        "actor": "7",
    }
    values.update(overrides)
    return PendingApproval(**values)


def _inbound(text: str, **overrides) -> InboundMessage:
    values: dict[str, Any] = {
        "channel": "telegram",
        "sender_id": "7",
        "chat_id": "42",
        "content": text,
    }
    values.update(overrides)
    return InboundMessage(**values)


def test_render_request_names_the_action_and_the_reply_words():
    text = render_request("cancel_order", "PO-1")
    assert "cancel_order" in text
    assert "PO-1" in text
    assert "yes" in text and "no" in text


def test_request_is_resolved_by_a_chat_reply():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        pending = _pending()
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(broker.request(pending))
        )
        worker.start()
        prompt = await asyncio.wait_for(bus.next_outbound(), 2)
        consumed = broker.consume(_inbound("yes"))
        await _join(worker)
        return prompt, consumed, approved

    prompt, consumed, approved = asyncio.run(scenario())
    assert prompt.metadata["kind"] == "approval"
    assert prompt.metadata["approval_id"]
    assert prompt.buttons == [["Approve", "Reject"]]
    assert "Approval required" in prompt.content
    assert consumed is True
    assert approved == [True]


def test_consume_refuses_a_reply_from_another_person():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(broker.request(_pending()))
        )
        worker.start()
        await asyncio.wait_for(bus.next_outbound(), 2)
        consumed = broker.consume(_inbound("yes", sender_id="8"))
        broker.cancel_all()
        await _join(worker)
        return consumed, approved

    consumed, approved = asyncio.run(scenario())
    assert consumed is False
    assert approved == [False]


def test_approval_prompt_keeps_the_forum_topic():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        gate = broker.gate(
            _inbound(
                "cancel it",
                metadata={"message_id": 11, "message_thread_id": 9},
            )
        )
        worker = threading.Thread(
            target=lambda: gate.request("cancel_order", "PO-1")
        )
        worker.start()
        prompt = await asyncio.wait_for(bus.next_outbound(), 2)
        broker.cancel_all()
        await _join(worker)
        return prompt

    prompt = asyncio.run(scenario())
    assert prompt.metadata["message_thread_id"] == 9


def test_callback_from_a_topic_without_a_session_key_still_resolves():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(
                broker.request(_pending(session_key="telegram:42:9"))
            )
        )
        worker.start()
        prompt = await asyncio.wait_for(bus.next_outbound(), 2)
        token = prompt.metadata["approval_id"]
        consumed = broker.consume(
            _inbound(
                "yes",
                chat_id="42",
                metadata={"approval_id": token},
            )
        )
        await _join(worker)
        return consumed, approved

    consumed, approved = asyncio.run(scenario())
    assert consumed is True
    assert approved == [True]


def test_consume_leaves_a_follow_up_question_for_the_agent():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(broker.request(_pending()))
        )
        worker.start()
        await asyncio.wait_for(bus.next_outbound(), 2)
        consumed = broker.consume(_inbound("what is this action?"))
        broker.cancel_all()
        await _join(worker)
        return consumed, approved

    consumed, approved = asyncio.run(scenario())
    assert consumed is False
    assert approved == [False]


def test_request_times_out_with_a_denial():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=0.05)
        broker.bind(asyncio.get_running_loop())
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(broker.request(_pending()))
        )
        worker.start()
        await asyncio.wait_for(bus.next_outbound(), 2)
        await _join(worker)
        return approved

    assert asyncio.run(scenario()) == [False]


def test_matching_callback_token_resolves_the_request():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(broker.request(_pending()))
        )
        worker.start()
        prompt = await asyncio.wait_for(bus.next_outbound(), 2)
        token = prompt.metadata["approval_id"]
        consumed = broker.consume(
            _inbound("yes", metadata={"approval_id": token})
        )
        await _join(worker)
        return consumed, approved

    consumed, approved = asyncio.run(scenario())
    assert consumed is True
    assert approved == [True]


def test_callback_from_an_older_prompt_is_ignored():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(broker.request(_pending()))
        )
        worker.start()
        await asyncio.wait_for(bus.next_outbound(), 2)
        consumed = broker.consume(
            _inbound("yes", metadata={"approval_id": "older"})
        )
        broker.cancel_all()
        await _join(worker)
        return consumed, approved

    consumed, approved = asyncio.run(scenario())
    assert consumed is False
    assert approved == [False]


def test_second_request_for_the_same_session_fails_closed():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=1.0)
        broker.bind(asyncio.get_running_loop())
        outcome: list[bool] = []
        worker = threading.Thread(
            target=lambda: outcome.append(broker.request(_pending()))
        )
        worker.start()
        await asyncio.wait_for(bus.next_outbound(), 2)
        outcome.append(broker.request(_pending(action="expedite_order")))
        broker.cancel_all()
        await _join(worker)
        return outcome

    assert asyncio.run(scenario()) == [False, False]


def test_cancel_denies_only_that_session_s_pending_request():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        outcome: list[bool] = []
        worker = threading.Thread(
            target=lambda: outcome.append(broker.request(_pending()))
        )
        worker.start()
        await asyncio.wait_for(bus.next_outbound(), 2)
        cancelled = broker.cancel("telegram:42")
        repeat = broker.cancel("telegram:42")
        await _join(worker)
        return cancelled, repeat, outcome

    assert asyncio.run(scenario()) == (True, False, [False])


def test_gate_audits_the_actor_and_decision():
    async def scenario():
        bus = MessageBus()
        broker = ApprovalBroker(bus, timeout=5.0)
        broker.bind(asyncio.get_running_loop())
        gate = broker.gate(_inbound("cancel it", metadata={"message_id": 11}))
        approved: list[bool] = []
        worker = threading.Thread(
            target=lambda: approved.append(gate.request("cancel_order", "PO-1"))
        )
        worker.start()
        prompt = await asyncio.wait_for(bus.next_outbound(), 2)
        broker.consume(_inbound("no"))
        await _join(worker)
        return prompt, gate, approved

    prompt, gate, approved = asyncio.run(scenario())
    assert isinstance(gate, ApprovalGate)
    assert prompt.reply_to == "11"
    assert approved == [False]
    assert gate.audit[-1].decision == "rejected"
    assert gate.audit[-1].actor == "7"
    assert gate.audit[-1].action == "cancel_order"


def test_broker_rejects_a_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        ApprovalBroker(MessageBus(), timeout=0)
