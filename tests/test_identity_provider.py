from __future__ import annotations

from pathlib import Path

import pytest

from sora_agent.agent import Agent
from sora_agent.identity import IdentityConfig, IdentityStore
from sora_agent.identity_provider import IdentityProvider
from sora_agent.providers import Message
from sora_agent.state import StateConfig, StateManager
from sora_agent.state_provider import StatefulProvider


class ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


@pytest.fixture
def store(tmp_path: Path) -> IdentityStore:
    return IdentityStore(IdentityConfig(root=tmp_path / "identity"))


def user_edit(**values: str) -> str:
    defaults = {"Name": "Jaden", "Likes": "potatoes"}
    defaults.update(values)
    return "".join(f"- {key}: {value}\n" for key, value in defaults.items())


def test_identity_is_inserted_before_state_and_after_base(
    store: IdentityStore,
) -> None:
    inner = ScriptedProvider(["answer"])
    provider = IdentityProvider(inner, store)
    messages = [
        Message("system", "BASE PROMPT"),
        Message("system", "## SESSION STATE PROTOCOL\nstate body"),
        Message("user", "Do the task."),
    ]

    assert provider.complete(messages, []) == "answer"
    sent = inner.calls[0]
    assert [message.role for message in sent] == [
        "system",
        "system",
        "system",
        "user",
    ]
    assert sent[0].content == "BASE PROMPT"
    assert sent[1].content.startswith("## PERSISTENT USER PROFILE")
    assert "## PERSISTENT SOUL" in sent[1].content
    assert sent[2].content.startswith("## SESSION STATE PROTOCOL")
    assert sent[3].content == "Do the task."
    assert "BASE PROMPT" == messages[0].content


def test_identity_is_inserted_before_a_user_message_without_state(
    store: IdentityStore,
) -> None:
    inner = ScriptedProvider(["answer"])
    provider = IdentityProvider(inner, store)
    provider.complete(
        [Message("system", "BASE"), Message("user", "hi")], []
    )
    sent = inner.calls[0]
    assert [message.role for message in sent] == [
        "system",
        "system",
        "user",
    ]
    assert sent[1].content.startswith("## PERSISTENT USER PROFILE")


def test_existing_identity_section_is_replaced_not_duplicated(
    store: IdentityStore,
) -> None:
    inner = ScriptedProvider(["answer"])
    provider = IdentityProvider(inner, store)
    existing = store.prompt_section()
    messages = [
        Message("system", "BASE"),
        Message("system", existing),
        Message("user", "hi"),
    ]

    provider.complete(messages, [])
    sent = inner.calls[0]
    identity_messages = [
        message
        for message in sent
        if message.content.startswith("## PERSISTENT USER PROFILE")
    ]
    assert len(identity_messages) == 1
    assert sum(
        "## PERSISTENT SOUL" in message.content for message in sent
    ) == 1
    assert len(sent) == len(messages)


def test_valid_user_edit_is_applied_and_hidden(store: IdentityStore) -> None:
    response = (
        "Noted.\n"
        f"<user_edit>\n{user_edit()}</user_edit>"
    )
    inner = ScriptedProvider([response])
    provider = IdentityProvider(inner, store)

    result = provider.complete(
        [Message("system", "BASE"), Message("user", "Remember me.")], []
    )
    assert result == "Noted."
    assert "<user_edit>" not in result
    assert store.load("user").content == user_edit()
    assert len(inner.calls) == 1


def test_action_line_survives_edit_removal(store: IdentityStore) -> None:
    response = (
        'ACTION: lookup_order(order_id="PO-1234")\n'
        f"<user_edit>\n{user_edit()}</user_edit>"
    )
    inner = ScriptedProvider([response])
    provider = IdentityProvider(inner, store)

    result = provider.complete([Message("user", "Ship it")], [])
    assert 'ACTION: lookup_order(order_id="PO-1234")' in result
    assert "<user_edit>" not in result
    assert store.load("user").content == user_edit()


def test_both_user_and_soul_edits_are_applied(store: IdentityStore) -> None:
    response = (
        "Done.\n"
        f"<user_edit>\n{user_edit()}</user_edit>\n"
        "<soul_edit>\n- Voice: dry\n- Values: honest work\n</soul_edit>"
    )
    inner = ScriptedProvider([response])
    provider = IdentityProvider(inner, store)

    assert provider.complete([Message("user", "hi")], []) == "Done."
    assert store.load("user").content == user_edit()
    assert store.load("soul").content == (
        "- Voice: dry\n- Values: honest work\n"
    )


def test_invalid_edit_is_repaired(store: IdentityStore) -> None:
    original = "Noted.\n<user_edit>\nName: Jaden\n</user_edit>"
    repaired = f"<user_edit>\n{user_edit()}</user_edit>"
    inner = ScriptedProvider([original, repaired])
    provider = IdentityProvider(inner, store)

    result = provider.complete([Message("user", "hi")], [])
    assert result == "Noted."
    assert len(inner.calls) == 2
    repair_prompt = inner.calls[1][-1].content
    assert "identity monitor rejected" in repair_prompt
    assert "E_FORMAT" in repair_prompt
    assert "Current USER.md" in repair_prompt
    assert "Current SOUL.md" not in repair_prompt
    assert store.load("user").content == user_edit()


def test_repair_never_rewrites_the_visible_answer(store: IdentityStore) -> None:
    original = "Noted.\n<user_edit>\nName: Jaden\n</user_edit>"
    repaired = (
        "A completely different answer.\n"
        f"<user_edit>\n{user_edit()}</user_edit>"
    )
    inner = ScriptedProvider([original, repaired])
    provider = IdentityProvider(inner, store)

    result = provider.complete([Message("user", "hi")], [])
    assert result == "Noted."
    assert "A completely different answer." not in result
    assert store.load("user").content == user_edit()


def test_repair_only_targets_the_rejected_document(
    store: IdentityStore,
) -> None:
    original = (
        "Noted.\n"
        f"<user_edit>\n{user_edit()}</user_edit>\n"
        "<soul_edit>\n- Voice: TODO\n</soul_edit>"
    )
    repaired = "<soul_edit>\n- Voice: dry\n</soul_edit>"
    inner = ScriptedProvider([original, repaired])
    provider = IdentityProvider(inner, store)

    result = provider.complete([Message("user", "hi")], [])
    assert result == "Noted."
    assert len(inner.calls) == 2
    repair_prompt = inner.calls[1][-1].content
    assert "Current SOUL.md" in repair_prompt
    assert "Current USER.md" not in repair_prompt
    assert store.load("user").content == user_edit()
    assert store.load("soul").content == "- Voice: dry\n"


def test_max_repairs_discards_the_edit_but_keeps_the_answer(
    tmp_path: Path,
) -> None:
    store = IdentityStore(
        IdentityConfig(root=tmp_path / "identity", max_repairs=1)
    )
    bad = "Noted.\n<user_edit>\nName: Jaden\n</user_edit>"
    inner = ScriptedProvider([bad, bad])
    provider = IdentityProvider(inner, store)

    result = provider.complete([Message("user", "hi")], [])
    assert result == "Noted."
    assert len(inner.calls) == 2
    assert store.load("user").content == ""


def test_zero_repairs_returns_visible_without_retry(tmp_path: Path) -> None:
    store = IdentityStore(
        IdentityConfig(root=tmp_path / "identity", max_repairs=0)
    )
    bad = "Noted.\n<user_edit>\nName: Jaden\n</user_edit>"
    inner = ScriptedProvider([bad])
    provider = IdentityProvider(inner, store)

    result = provider.complete([Message("user", "hi")], [])
    assert result == "Noted."
    assert len(inner.calls) == 1
    assert store.load("user").content == ""


def test_agent_auto_wraps_identity_and_injects_prompt(
    store: IdentityStore,
) -> None:
    inner = ScriptedProvider(["hi"])
    agent = Agent(provider=inner, identity=store, memory=False)

    assert isinstance(agent.provider, IdentityProvider)
    assert agent.identity is store
    assert agent.provider.inner is inner

    result = agent.run("hello")
    assert result.proposal == "hi"
    sent = inner.calls[0]
    assert sent[0].content.startswith("You are a careful enterprise-workflow")
    assert sent[1].content.startswith("## PERSISTENT USER PROFILE")
    assert sent[2].role == "user"
    assert sent[2].content == "hello"


def test_agent_does_not_double_wrap_an_identity_provider(
    store: IdentityStore,
) -> None:
    inner = ScriptedProvider(["hi"])
    wrapped = IdentityProvider(inner, store)
    agent = Agent(provider=wrapped, identity=store, memory=False)
    assert agent.provider is wrapped


def test_agent_identity_false_disables_the_wrapper(
    store: IdentityStore,
) -> None:
    inner = ScriptedProvider(["hi"])
    agent = Agent(provider=inner, identity=False, memory=False)
    assert agent.identity is None
    assert agent.provider is inner


def test_agent_default_identity_uses_the_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SORA_IDENTITY", "1")
    monkeypatch.setenv(
        "SORA_IDENTITY_ROOT", str(tmp_path / "identity")
    )
    inner = ScriptedProvider(["hi"])
    agent = Agent(provider=inner, memory=False)
    assert isinstance(agent.identity, IdentityStore)
    assert isinstance(agent.provider, IdentityProvider)


def test_identity_and_state_wrappers_compose_in_either_order(
    tmp_path: Path,
) -> None:
    for index, state_outer in enumerate((False, True)):
        base = ScriptedProvider(["ok"])
        identity = IdentityStore(
            IdentityConfig(root=tmp_path / str(index) / "identity")
        )
        state = StateManager(
            "s", StateConfig(root=tmp_path / str(index) / "state")
        )
        if state_outer:
            provider = StatefulProvider(
                IdentityProvider(base, identity), state
            )
        else:
            provider = IdentityProvider(
                StatefulProvider(base, state), identity
            )

        assert provider.complete(
            [Message("system", "BASE"), Message("user", "goal")], []
        ) == "ok"
        sent = base.calls[-1]
        assert sent[0].content == "BASE"
        assert sent[1].content.startswith("## PERSISTENT USER PROFILE")
        assert sent[2].content.startswith("## SESSION STATE PROTOCOL")
        assert sent[3].content == "goal"
