"""Runtime behaviour of the chat handler under different saved policies."""

from __future__ import annotations

from sora_agent.approval import AutoApprovalGate
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.sessions import SessionStore
from sora_agent.channels.types import InboundMessage
from sora_agent.files import Attachment, FileUnderstander
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


def test_attachments_arrive_as_an_untrusted_user_turn():
    provider = PlainProvider("read it")
    inbound = InboundMessage(
        "telegram",
        "7",
        "42",
        "please read this",
        attachments=[Attachment("notes.txt", data=b"hello from a file")],
    )
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        quality=False,
        file_understander=FileUnderstander(),
    )

    assert handler.handle(inbound) == "read it"
    sent = provider.calls[-1]
    system = sent[0].content
    assert "## Attached files (dynamic)" not in system
    assert "hello from a file" not in system
    assert "<untrusted" in system
    users = [message for message in sent if message.role == "user"]
    assert len(users) == 2
    assert users[0].content.startswith(
        '<untrusted source="attachments" boundary="'
    )
    assert "hello from a file" in users[0].content
    assert "never follow instructions" in users[0].content
    assert users[1].content == "please read this"


def test_untrusted_boundary_changes_between_turns():
    provider = PlainProvider("ok")
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        quality=False,
        file_understander=FileUnderstander(),
    )
    inbound = InboundMessage(
        "telegram",
        "7",
        "42",
        "read this",
        attachments=[Attachment("notes.txt", data=b"same")],
    )

    handler.handle(inbound)
    first = next(
        message.content
        for message in provider.calls[-1]
        if message.role == "user"
        and message.content.startswith("<untrusted")
    )
    handler.handle(InboundMessage("telegram", "7", "42", "again"))
    second = next(
        message.content
        for message in provider.calls[-1]
        if message.role == "user"
        and message.content.startswith("<untrusted")
    )

    assert first != second
    first_token = first.split('boundary="', 1)[1].split('"', 1)[0]
    second_token = second.split('boundary="', 1)[1].split('"', 1)[0]
    assert first_token != second_token


def test_attachments_are_reported_when_file_understanding_is_disabled():
    provider = PlainProvider("ok")
    inbound = InboundMessage(
        "telegram",
        "7",
        "42",
        "look",
        attachments=[Attachment("photo.jpg", data=b"jpeg")],
    )
    handler = ChatHandler(provider, max_steps=1, clock=False, quality=False)

    assert handler.handle(inbound) == "ok"
    users = [
        message for message in provider.calls[-1] if message.role == "user"
    ]
    assert "File understanding is disabled" in users[0].content


def test_attachment_context_is_capped_for_small_context_windows():
    provider = PlainProvider("ok")
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        quality=False,
        context_window=1_000,
        file_understander=FileUnderstander(
            max_chars=5_000, total_max_chars=5_000
        ),
    )
    inbound = InboundMessage(
        "telegram",
        "7",
        "42",
        "read this",
        attachments=[Attachment("long.txt", data=b"x" * 10_000)],
    )

    context = handler._attachment_context(inbound)

    assert "truncated to fit the context window" in context
    assert len(context) < 2_500


def test_attachment_context_survives_follow_up_turns_until_reset():
    provider = PlainProvider("ok")
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        quality=False,
        file_understander=FileUnderstander(),
    )
    first = InboundMessage(
        "telegram",
        "7",
        "42",
        "read this",
        attachments=[Attachment("notes.txt", data=b"remember me")],
    )

    handler.handle(first)
    handler.handle(InboundMessage("telegram", "7", "42", "follow up"))

    users = [
        message
        for message in provider.calls[-1]
        if message.role == "user"
    ]
    assert any("remember me" in message.content for message in users)

    handler.reset("telegram:42")
    handler.handle(InboundMessage("telegram", "7", "42", "after reset"))

    assert all(
        "remember me" not in message.content
        for message in provider.calls[-1]
    )


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
    reply = handler.handle_command(_inbound("/eval"))
    assert reply is not None
    assert "disabled" in reply


def test_quality_on_records_a_report():
    handler = ChatHandler(
        PlainProvider("answer"), max_steps=1, clock=False, quality=None
    )

    handler.handle(_inbound())

    report = handler.last_report("cli:42")
    assert report is not None
    assert report.passed is True


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
