"""Runtime behaviour of the chat handler under different saved policies."""

from __future__ import annotations

from sora_agent.approval import AutoApprovalGate
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.sessions import SessionStore
from sora_agent.channels.types import (
    STREAM_DELTA_KIND,
    STREAM_END_KIND,
    InboundMessage,
    TurnReply,
)
from sora_agent.providers import Message, ModelTurn, ToolCall
from sora_agent.tools import Tool, ToolRegistry


class PlainProvider:
    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        return ModelTurn(text=self.text)


class ApprovalProvider:
    def complete_turn(self, messages, tools) -> ModelTurn:
        if any(message.role == "tool" for message in messages):
            return ModelTurn(text="done")
        return ModelTurn(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="expedite_order",
                    arguments={"order_id": "PO-1"},
                )
            ]
        )


def _inbound(text: str = "hi", chat: str = "42") -> InboundMessage:
    return InboundMessage("cli", "local", chat, text)


def _approval_registry(executed: list[str]) -> ToolRegistry:
    def expedite(order_id: str) -> str:
        executed.append(order_id)
        return "expedited"

    registry = ToolRegistry()
    registry.add(
        Tool(
            expedite,
            "expedite_order",
            "Expedite an order.",
            needs_approval=True,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Approval off
# ---------------------------------------------------------------------------


def test_approval_off_executes_and_still_writes_an_audit_trail():
    executed: list[str] = []
    gate = AutoApprovalGate()
    handler = ChatHandler(
        ApprovalProvider(),
        _approval_registry(executed),
        approval_gate=gate,
        max_steps=2,
        clock=False,
    )

    assert handler.handle(_inbound("expedite PO-1")) == "done"

    assert executed == ["PO-1"]
    assert len(gate.audit) == 2
    decision, execution = gate.audit
    assert decision.action == "expedite_order"
    assert decision.actor == "policy:approval-off"
    assert decision.decision == "auto-approved"
    assert "PO-1" in decision.detail
    assert execution.decision == "executed"


# ---------------------------------------------------------------------------
# Quality off
# ---------------------------------------------------------------------------


def test_quality_off_skips_evaluation_but_still_runs_the_turn():
    handler = ChatHandler(
        PlainProvider("answer"), max_steps=1, clock=False, quality=False
    )

    assert handler.handle(_inbound()) == "answer"
    assert handler.quality is None
    assert handler.last_report("cli:42") is None
    assert "disabled" in handler.eval_report("cli:42")
    assert "disabled" in handler.handle_command(_inbound("/eval"))


def test_quality_on_records_a_report():
    handler = ChatHandler(
        PlainProvider("answer"), max_steps=1, clock=False, quality=None
    )

    handler.handle(_inbound())

    report = handler.last_report("cli:42")
    assert report is not None
    assert report.passed is True


# ---------------------------------------------------------------------------
# Streaming off
# ---------------------------------------------------------------------------


def test_streaming_off_never_emits_live_updates():
    events: list = []
    handler = ChatHandler(
        PlainProvider("answer"),
        max_steps=1,
        clock=False,
        quality=False,
        emitter=events.append,
        streaming=False,
    )

    reply = handler.handle(_inbound())

    assert reply == "answer"
    assert events == []


def test_streaming_on_emits_cumulative_snapshots_and_ends():
    events: list = []
    handler = ChatHandler(
        PlainProvider("streamed answer"),
        max_steps=1,
        clock=False,
        quality=False,
        emitter=events.append,
        streaming=True,
    )

    reply = handler.handle(_inbound())

    assert isinstance(reply, TurnReply)
    assert reply.delivered is True
    kinds = [event.metadata["kind"] for event in events]
    assert STREAM_DELTA_KIND in kinds
    assert kinds[-1] == STREAM_END_KIND
    assert any(event.content == "streamed answer" for event in events)


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


def test_handler_trims_persisted_history_to_the_context_window(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    key = "cli:42"
    history = [
        Message(
            "user" if index % 2 == 0 else "assistant",
            f"message {index}: " + "x" * 600,
        )
        for index in range(30)
    ]
    store.save(key, history)

    provider = PlainProvider("ok")
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        quality=False,
        sessions=store,
        context_window=2_000,
    )

    handler.handle(_inbound("current question"))

    sent = provider.calls[0]
    assert sent[0].role == "system"
    assert sent[-1].content == "current question"
    # The 30 persisted messages no longer all fit, so older ones were dropped.
    assert len(sent) < len(history) + 2


def test_handler_keeps_full_history_when_no_window_is_configured(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    key = "cli:42"
    history = [
        Message("user", f"message {index}: " + "x" * 600)
        for index in range(30)
    ]
    store.save(key, history)

    provider = PlainProvider("ok")
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        quality=False,
        sessions=store,
        context_window=None,
    )

    handler.handle(_inbound("current question"))

    assert len(provider.calls[0]) == len(history) + 2
