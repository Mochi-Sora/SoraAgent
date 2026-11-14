"""Streaming must never expose internal identity edit blocks."""

from __future__ import annotations

from pathlib import Path

import pytest

from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.types import STREAM_DELTA_KIND, InboundMessage
from sora_agent.identity import IdentityConfig, IdentityStore
from sora_agent.identity_provider import IdentityProvider
from sora_agent.providers import Message, ModelTurn


class ChunkedProvider:
    """Stream a fixed response in chunks, like a provider adapter."""

    def __init__(self, text: str, size: int = 5) -> None:
        self.text = text
        self.size = size
        self.calls = 0

    def stream_turn(self, messages, tools, on_delta, *, should_stop=None):
        self.calls += 1
        for index in range(0, len(self.text), self.size):
            if should_stop is not None and should_stop():
                break
            on_delta(self.text[index : index + self.size])
        return ModelTurn(text=self.text)

    def complete(self, messages, tools) -> str:
        self.calls += 1
        return self.text


@pytest.fixture
def store(tmp_path: Path) -> IdentityStore:
    return IdentityStore(IdentityConfig(root=tmp_path / "identity"))


def _inbound(text: str = "hi") -> InboundMessage:
    return InboundMessage("desktop", "desktop", "session", text)


def test_streamed_answer_hides_the_edit_block(store: IdentityStore) -> None:
    raw = (
        "Nice to meet you.\n\n"
        "<user_edit>\n- Name: Jaden\n- Likes: potatoes\n</user_edit>\n\n"
        "What should we do next?"
    )
    deltas: list[str] = []
    provider = IdentityProvider(ChunkedProvider(raw, size=4), store)

    turn = provider.stream_turn([Message("user", "hi")], [], deltas.append)
    streamed = "".join(deltas)

    assert "<user_edit>" not in streamed
    assert "- Name: Jaden" not in streamed
    assert streamed.strip() == turn.text
    assert turn.text == "Nice to meet you.\n\nWhat should we do next?"
    assert store.load("user").content == "- Name: Jaden\n- Likes: potatoes\n"


def test_streamed_answer_hides_a_split_tag(store: IdentityStore) -> None:
    raw = (
        "Hello <us"
        "er_edit>\n- Name: Jaden\n</us"
        "er_edit> world"
    )
    deltas: list[str] = []
    provider = IdentityProvider(ChunkedProvider(raw, size=3), store)

    turn = provider.stream_turn([Message("user", "hi")], [], deltas.append)
    streamed = "".join(deltas)

    assert "user_edit" not in streamed
    assert streamed.strip() == "Hello  world"
    assert streamed.strip() == turn.text


def test_streamed_answer_hides_soul_edits(store: IdentityStore) -> None:
    raw = (
        "Done.\n"
        "<soul_edit>\n- Voice: dry\n- Values: honest work\n</soul_edit>\n"
        "Anything else?"
    )
    deltas: list[str] = []
    provider = IdentityProvider(ChunkedProvider(raw, size=2), store)

    turn = provider.stream_turn([Message("user", "hi")], [], deltas.append)
    streamed = "".join(deltas)

    assert "<soul_edit>" not in streamed
    assert streamed.strip() == turn.text
    assert store.load("soul").content == "- Voice: dry\n- Values: honest work\n"


def test_handler_does_not_append_a_second_message_when_the_edit_is_hidden(
    store: IdentityStore,
) -> None:
    raw = (
        "Noted.\n\n"
        "<user_edit>\n- Name: Jaden\n- Likes: potatoes\n</user_edit>\n\n"
        "A follow-up question."
    )
    emitted = []
    provider = IdentityProvider(ChunkedProvider(raw, size=5), store)
    handler = ChatHandler(provider, max_steps=1, emitter=emitted.append)

    reply = handler.handle(_inbound())

    assert reply.delivered is True
    deltas = [
        message.content
        for message in emitted
        if message.metadata.get("kind") == STREAM_DELTA_KIND
    ]
    assert deltas
    assert deltas[-1].strip() == reply.text.strip()
    assert "<user_edit>" not in deltas[-1]


def test_unclosed_edit_block_is_not_streamed(store: IdentityStore) -> None:
    raw = "Answer.\n<user_edit>\n- Name: Jaden\n- Likes: potatoes\n"
    deltas: list[str] = []
    provider = IdentityProvider(ChunkedProvider(raw, size=4), store)

    turn = provider.stream_turn([Message("user", "hi")], [], deltas.append)
    streamed = "".join(deltas)

    assert "<user_edit>" not in streamed
    assert "- Name: Jaden" not in streamed
    assert streamed.strip() == "Answer."
    assert streamed.strip() == turn.text


def test_streamed_edit_with_duplicate_keys_is_still_hidden(
    store: IdentityStore,
) -> None:
    # The model sometimes emits two "- Likes:" lines.  The edit is rejected
    # and repaired, but the raw block must not appear in the live stream.
    raw = (
        "Noted.\n\n"
        "<user_edit>\n- Name: Jaden\n- Likes: potatoes\n"
        "- Likes: badminton\n</user_edit>\n\n"
        "A follow-up question."
    )
    deltas: list[str] = []
    provider = IdentityProvider(ChunkedProvider(raw, size=3), store)

    turn = provider.stream_turn([Message("user", "hi")], [], deltas.append)
    streamed = "".join(deltas)

    assert "<user_edit>" not in streamed
    assert "- Likes:" not in streamed
    assert streamed.strip() == turn.text
    assert store.load("user").content == ""
