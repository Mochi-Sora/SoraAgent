"""Tests for the deterministic context-window budget."""

from __future__ import annotations

from sora_agent.context import estimate_tokens, trim_messages
from sora_agent.providers import Message, ToolCall


def _contents(messages):
    return [message.content for message in messages]


def test_estimate_tokens_is_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_trim_keeps_the_newest_messages_when_they_fit():
    messages = [Message("user", "old"), Message("assistant", "older reply")]

    kept = trim_messages(messages, 10_000)

    assert _contents(kept) == ["old", "older reply"]


def test_trim_stops_at_the_first_message_that_does_not_fit():
    messages = [
        Message("user", "x" * 6_000),
        Message("assistant", "recent"),
    ]

    kept = trim_messages(messages, 1_100, reserve_tokens=0, fixed_texts=("s",))

    assert _contents(kept) == ["recent"]


def test_trim_always_keeps_and_truncates_the_newest_message():
    messages = [Message("user", "x" * 10_000)]

    kept = trim_messages(messages, 500, reserve_tokens=0)

    assert len(kept) == 1
    assert kept[0].content.startswith("x")
    assert kept[0].content.endswith("[...truncated to fit context]")
    assert len(kept[0].content) < 10_000


def test_trim_accounts_for_fixed_system_and_user_text():
    messages = [Message("user", "x" * 3_000)]

    kept = trim_messages(
        messages,
        600,
        reserve_tokens=0,
        fixed_texts=("system prompt", "current question"),
    )

    assert len(kept) == 1
    assert kept[0].content.endswith("[...truncated to fit context]")


def test_trim_preserves_tool_call_metadata():
    call = ToolCall(id="c1", name="lookup_order", arguments={"order_id": "PO-1"})
    messages = [
        Message("assistant", "", tool_calls=[call]),
        Message("tool", "result", tool_call_id="c1", name="lookup_order"),
    ]

    kept = trim_messages(messages, 10_000)

    assert kept[0].tool_calls == [call]
    assert kept[1].tool_call_id == "c1"


def test_trim_returns_an_independent_list():
    messages = [Message("user", "hello")]

    kept = trim_messages(messages, 10_000)
    kept.append(Message("assistant", "extra"))

    assert len(messages) == 1


def test_non_positive_budget_is_a_noop():
    messages = [Message("user", "hello")]

    assert trim_messages(messages, 0) == messages
    assert trim_messages([], 1_000) == []


def test_history_is_trimmed_to_the_configured_window():
    messages = [
        Message("user", f"old message {index}: " + "x" * 200)
        for index in range(20)
    ]

    kept = trim_messages(
        messages,
        800,
        reserve_tokens=0,
        fixed_texts=("system",),
    )

    assert kept
    assert kept[-1] is messages[-1]
    # The oldest messages are dropped, so the retained block is contiguous
    # and ends at the newest message.
    start = messages.index(kept[0])
    assert messages[start:] == kept
