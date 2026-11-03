"""Tests for the model-facing clock and system-prompt ordering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sora_agent.agent import Agent
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.types import InboundMessage
from sora_agent.clock import (
    CLOCK_HEADING,
    ClockProvider,
    append_clock,
    clock_section,
    extract_clock,
    strip_clock,
    with_clock,
    wrap_clock,
)
from sora_agent.identity import IdentityConfig, IdentityStore
from sora_agent.identity_provider import IdentityProvider
from sora_agent.providers import Message, ModelTurn
from sora_agent.state import StateConfig, StateManager
from sora_agent.state_provider import StatefulProvider

NOW = datetime(2026, 11, 3, 14, 5, 12, tzinfo=timezone.utc)


class ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


class TurnProvider:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []
        self.tools: list[list[str]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        self.tools.append([tool.name for tool in tools])
        return self.turns.pop(0)


def _inbound(text: str) -> InboundMessage:
    return InboundMessage(channel="fake", chat_id="42", sender_id="7", content=text)


def test_clock_section_has_local_utc_and_weekday() -> None:
    section = clock_section(NOW)

    assert section.startswith(CLOCK_HEADING)
    assert "Local: 2026-11-03T14:05:12+00:00" in section
    assert "Tuesday" in section
    assert "UTC: 2026-11-03T14:05:12Z" in section
    assert "authoritative current time" in section


def test_clock_section_interprets_naive_time_as_local() -> None:
    section = clock_section(datetime(2026, 11, 3, 9, 30))

    assert "Local: 2026-11-03T09:30:00" in section


def test_extract_strip_and_append_clock_round_trip() -> None:
    messages = [
        Message("system", "BASE"),
        Message("system", clock_section(NOW)),
        Message("user", "hi"),
    ]

    assert extract_clock(messages) == clock_section(NOW)
    stripped = strip_clock(messages)
    assert [message.role for message in stripped] == ["system", "user"]
    assert stripped[0].content == "BASE"

    rebuilt = append_clock(stripped, clock_section(NOW))
    assert [message.role for message in rebuilt] == ["system", "system", "user"]
    assert rebuilt[1].content.startswith(CLOCK_HEADING)


def test_with_clock_replaces_an_existing_section() -> None:
    older = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
    messages = append_clock([Message("system", "BASE")], clock_section(older))

    rebuilt = with_clock(messages, NOW)

    assert len(rebuilt) == 2
    assert "14:05:12" in rebuilt[1].content
    assert "09:00:00" not in rebuilt[1].content


def test_clock_stays_behind_state_and_identity_in_either_wrapper_order(
    tmp_path: Path,
) -> None:
    for index, state_outer in enumerate((False, True)):
        inner = ScriptedProvider(["ok"])
        identity = IdentityStore(
            IdentityConfig(root=tmp_path / str(index) / "identity")
        )
        state = StateManager(
            "s", StateConfig(root=tmp_path / str(index) / "state")
        )
        clocked = ClockProvider(inner, NOW)
        if state_outer:
            provider = StatefulProvider(
                IdentityProvider(clocked, identity), state
            )
        else:
            provider = IdentityProvider(
                StatefulProvider(clocked, state), identity
            )

        assert provider.complete(
            [Message("system", "BASE"), Message("user", "goal")], []
        ) == "ok"
        sent = inner.calls[-1]
        assert [message.role for message in sent] == [
            "system",
            "system",
            "system",
            "system",
            "user",
        ]
        assert sent[0].content == "BASE"
        assert sent[1].content.startswith("## PERSISTENT USER PROFILE")
        assert sent[2].content.startswith("## SESSION STATE PROTOCOL")
        assert sent[3].content.startswith(CLOCK_HEADING)


def test_clock_provider_refreshes_the_section_each_turn() -> None:
    moments = iter(
        [
            datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 3, 9, 1, tzinfo=timezone.utc),
        ]
    )
    inner = ScriptedProvider(["one", "two"])
    provider = ClockProvider(inner, lambda: next(moments))

    provider.complete([Message("system", "BASE"), Message("user", "a")], [])
    provider.complete([Message("system", "BASE"), Message("user", "b")], [])

    assert inner.calls[0][-2].content.startswith(CLOCK_HEADING)
    assert "09:00:00" in inner.calls[0][-2].content
    assert "09:01:00" in inner.calls[1][-2].content


def test_wrap_clock_is_idempotent_and_can_be_disabled() -> None:
    inner = ScriptedProvider(["ok"])
    wrapped = wrap_clock(inner, NOW)

    assert isinstance(wrapped, ClockProvider)
    assert wrap_clock(wrapped, NOW) is wrapped
    assert wrap_clock(inner, False) is inner


def test_agent_injects_clock_by_default_and_accepts_an_opt_out() -> None:
    provider = TurnProvider([ModelTurn(text="Done.")])
    agent = Agent(provider=provider, memory=False, identity=False, clock=NOW)

    agent.run("hello")

    system = [
        message for message in provider.calls[0] if message.role == "system"
    ]
    assert len(system) == 2
    assert system[0].content.startswith("You are a careful enterprise-workflow")
    assert system[1].content.startswith(CLOCK_HEADING)

    plain = TurnProvider([ModelTurn(text="Done.")])
    Agent(
        provider=plain,
        memory=False,
        identity=False,
        clock=False,
    ).run("hello")
    assert [message.role for message in plain.calls[0]] == ["system", "user"]


def test_agent_clock_lands_after_state_when_a_state_wrapper_is_used(
    tmp_path: Path,
) -> None:
    inner = TurnProvider([ModelTurn(text="Done.")])
    state = StateManager("s", StateConfig(root=tmp_path / "state"))
    agent = Agent(
        provider=StatefulProvider(inner, state),
        memory=False,
        identity=False,
        clock=NOW,
    )

    agent.run("goal")

    system = [m for m in inner.calls[0] if m.role == "system"]
    assert system[-2].content.startswith("## SESSION STATE PROTOCOL")
    assert system[-1].content.startswith(CLOCK_HEADING)


def test_chat_handler_injects_clock_by_default() -> None:
    provider = TurnProvider([ModelTurn(text="hello")])
    handler = ChatHandler(provider, max_steps=1, clock=NOW)

    assert handler.handle(_inbound("hi")) == "hello"

    system = [
        message for message in provider.calls[0] if message.role == "system"
    ]
    assert len(system) == 2
    assert system[1].content.startswith(CLOCK_HEADING)
