"""Tests for durable per-session conversation history."""

from __future__ import annotations

import json
import stat

from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.sessions import SessionStore
from sora_agent.channels.types import InboundMessage
from sora_agent.providers import Message, ModelTurn, ToolCall


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        return ModelTurn(text="ok")


def _inbound(text: str, chat: str = "42") -> InboundMessage:
    return InboundMessage("fake", "7", chat, text)


def _messages() -> list[Message]:
    return [
        Message("user", "hello ☃"),
        Message(
            "assistant",
            "checking",
            tool_calls=[ToolCall("c1", "lookup", {"id": 1})],
        ),
        Message("tool", "result", tool_call_id="c1", name="lookup"),
    ]


def test_round_trip_preserves_messages_and_tool_calls(tmp_path):
    store = SessionStore(tmp_path)
    store.save("telegram:42", _messages())

    loaded = store.load("telegram:42")

    assert [message.role for message in loaded] == ["user", "assistant", "tool"]
    assert loaded[0].content == "hello ☃"
    assert loaded[1].tool_calls[0].arguments == {"id": 1}
    assert loaded[2].tool_call_id == "c1"
    mode = stat.S_IMODE(store.path("telegram:42").stat().st_mode)
    assert mode == 0o600


def test_sessions_are_isolated(tmp_path):
    store = SessionStore(tmp_path)
    store.save("fake:1", [Message("user", "one")])
    store.save("fake:2", [Message("user", "two")])

    assert [m.content for m in store.load("fake:1")] == ["one"]
    assert [m.content for m in store.load("fake:2")] == ["two"]
    assert store.load("fake:3") == []


def test_clear_removes_the_file(tmp_path):
    store = SessionStore(tmp_path)
    store.save("fake:1", [Message("user", "one")])
    store.clear("fake:1")
    assert store.load("fake:1") == []
    store.clear("fake:1")  # idempotent


def test_corrupt_session_is_ignored(tmp_path):
    store = SessionStore(tmp_path)
    target = store.path("fake:1")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")
    assert store.load("fake:1") == []


def test_unknown_version_is_ignored(tmp_path):
    store = SessionStore(tmp_path)
    target = store.path("fake:1")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"version": 99, "messages": []}), encoding="utf-8"
    )
    assert store.load("fake:1") == []


def test_disabled_store_does_nothing(tmp_path):
    store = SessionStore(tmp_path, enabled=False)
    store.save("fake:1", [Message("user", "one")])
    assert store.load("fake:1") == []
    assert not store.path("fake:1").exists()


def test_chat_history_survives_a_new_handler(tmp_path):
    store = SessionStore(tmp_path)
    provider = RecordingProvider()

    first = ChatHandler(provider, max_steps=1, sessions=store)
    assert first.handle(_inbound("one")) == "ok"

    second = ChatHandler(provider, max_steps=1, sessions=store)
    assert second.handle(_inbound("two")) == "ok"

    roles = [message.role for message in provider.calls[-1]]
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert provider.calls[-1][2].content == "one"


def test_reset_clears_persisted_history(tmp_path):
    store = SessionStore(tmp_path)
    handler = ChatHandler(RecordingProvider(), max_steps=1, sessions=store)
    handler.handle(_inbound("one"))
    assert store.load("fake:42")

    assert handler.handle_command(_inbound("/reset")) == "Conversation reset."
    assert store.load("fake:42") == []


def test_history_is_trimmed_before_saving(tmp_path):
    store = SessionStore(tmp_path)
    handler = ChatHandler(
        RecordingProvider(), max_steps=1, max_turns=1, sessions=store
    )
    handler.handle(_inbound("one"))
    handler.handle(_inbound("two"))

    loaded = store.load("fake:42")
    assert [message.content for message in loaded] == ["two", "ok"]
