from __future__ import annotations

import json
from pathlib import Path

import pytest

from sora_agent.agent import Agent
from sora_agent.providers import Message
from sora_agent.state import (
    DEFAULT_MAX_REPAIRS,
    MAX_STATE_LIMIT,
    MIN_STATE_LIMIT,
    STATE_HEADINGS,
    StateConfig,
    StateManager,
    StateMonitor,
    canonicalize,
    extract_state_edit,
    validate_session_id,
)
from sora_agent.state_provider import StatefulProvider
from sora_agent.tools import Tool, ToolRegistry


def valid_state(
    goal: str = "Ship PO-1234 on time.",
    current: str = "No work done yet.",
    next_steps: str = "Confirm the ship date.",
    feeling: str = "Concerned.",
    thinking: str = "Wants speed.",
) -> str:
    return canonicalize(
        f"""## Goal
{goal}

## Current State
{current}

## {STATE_HEADINGS[2]}
{next_steps}

## User
Feeling: {feeling}
Thinking: {thinking}
"""
    )


class ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


@pytest.fixture
def config(tmp_path: Path) -> StateConfig:
    return StateConfig(root=tmp_path)


def test_config_clamps_limits(tmp_path: Path) -> None:
    assert StateConfig(root=tmp_path, limit=100).limit == MIN_STATE_LIMIT
    assert StateConfig(root=tmp_path, limit=99_999).limit == MAX_STATE_LIMIT
    assert StateConfig(root=tmp_path, limit=4076).limit == 4076
    assert StateConfig(root=tmp_path).max_repairs == DEFAULT_MAX_REPAIRS == 3
    assert StateConfig(root=tmp_path, max_repairs=0).max_repairs == 1
    assert StateConfig(root=tmp_path, max_repairs=99).max_repairs == 5
    with pytest.raises(TypeError):
        StateConfig(root=tmp_path, limit=4076.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StateConfig(root=tmp_path, max_repairs=True)  # type: ignore[arg-type]


def test_initial_state_is_valid_and_session_named(config: StateConfig) -> None:
    manager = StateManager("po-1234", config)
    content = manager.ensure("Ship PO-1234 on time.")
    assert manager.monitor.validate(content).ok
    assert manager.path.name == "po-1234.state.md"
    assert manager.path.parent == config.root
    assert manager.path.read_text(encoding="utf-8") == content


def test_ensure_never_overwrites_existing_state(config: StateConfig) -> None:
    manager = StateManager("s", config)
    first = manager.ensure("Original goal")
    second = manager.ensure("A completely different goal")
    assert second == first
    assert manager.path.read_text(encoding="utf-8") == first


def test_apply_writes_and_backs_up_previous(config: StateConfig) -> None:
    manager = StateManager("s", config)
    first = manager.ensure("Original goal")
    second = valid_state(current="Expedite approved.")
    result = manager.apply(second)
    assert result.status == "accepted"
    assert manager.path.read_text(encoding="utf-8") == second
    assert manager.backup_path.read_text(encoding="utf-8") == first


def test_applied_state_edits_are_audited(config: StateConfig) -> None:
    manager = StateManager("s", config)
    first = manager.ensure("Original goal")
    second = valid_state(current="Expedite approved.")

    assert manager.apply(second).status == "accepted"
    assert manager.apply(second).status == "unchanged"

    lines = manager.audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert all(record["event"] == "state_edit" for record in records)
    assert all(record["session"] == "s" for record in records)
    assert records[0]["new_sha256"] == records[1]["previous_sha256"]
    assert records[1]["chars"] == len(second)
    assert records[0]["previous_sha256"] != records[0]["new_sha256"]
    assert "at" in records[1]
    assert first != second


def test_rejected_edit_never_changes_file(config: StateConfig) -> None:
    manager = StateManager("s", config)
    first = manager.ensure("Original goal")
    result = manager.apply("## Goal\nTODO\n")
    assert result.status == "rejected"
    assert manager.path.read_text(encoding="utf-8") == first
    assert not manager.backup_path.exists()


def test_unchanged_edit_is_noop(config: StateConfig) -> None:
    manager = StateManager("s", config)
    first = manager.ensure("Original goal")
    result = manager.apply(first)
    assert result.status == "unchanged"
    assert manager.last_apply == result
    assert manager.path.read_text(encoding="utf-8") == first


def test_store_refuses_to_write_empty_state(config: StateConfig) -> None:
    manager = StateManager("s", config)
    with pytest.raises(ValueError):
        manager.store.write_atomic("s", "   \n")
    assert not manager.path.exists()


def test_missing_heading_rejected() -> None:
    text = valid_state().replace(
        f"## {STATE_HEADINGS[2]}\nConfirm the ship date.\n", ""
    )
    result = StateMonitor().validate(text)
    assert not result.ok
    assert any(
        issue.code == "E_HEADING" and STATE_HEADINGS[2] in issue.message
        for issue in result.issues
    )


def test_extra_heading_rejected() -> None:
    text = valid_state().replace(
        f"## {STATE_HEADINGS[2]}",
        f"## Notes\nsome notes\n\n## {STATE_HEADINGS[2]}",
    )
    result = StateMonitor().validate(text)
    assert not result.ok
    assert any(issue.code == "E_STRUCTURE" for issue in result.issues)


def test_user_requires_both_lines() -> None:
    text = valid_state().replace("Thinking: Wants speed.\n", "")
    result = StateMonitor().validate(text)
    assert not result.ok
    assert any(issue.code == "E_USER" for issue in result.issues)


def test_user_rejects_extra_lines() -> None:
    text = valid_state().replace(
        "Thinking: Wants speed.\n",
        "Thinking: Wants speed.\nExtra: nope.\n",
    )
    result = StateMonitor().validate(text)
    assert not result.ok
    assert any(issue.code == "E_USER" for issue in result.issues)


def test_over_limit_rejected() -> None:
    text = valid_state(current="x" * (MIN_STATE_LIMIT + 10))
    result = StateMonitor(MIN_STATE_LIMIT).validate(text)
    assert any(issue.code == "E_TOO_LONG" for issue in result.issues)


def test_gibberish_character_run_rejected() -> None:
    text = valid_state(goal="a" * 30)
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_GIBBERISH" for issue in result.issues)


def test_gibberish_token_repetition_rejected() -> None:
    text = valid_state(current="hello " * 20)
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_GIBBERISH" for issue in result.issues)


def test_placeholder_rejected() -> None:
    text = valid_state(next_steps="TODO")
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_PLACEHOLDER" for issue in result.issues)


def test_clearing_a_section_is_rejected() -> None:
    previous = valid_state(goal="Ship PO-1234 on time.")
    candidate = valid_state(goal="TODO")
    result = StateMonitor().validate(candidate, previous)
    assert any(issue.code == "E_CLEARED" for issue in result.issues)


def test_clearing_user_value_is_rejected() -> None:
    previous = valid_state(feeling="Concerned.", thinking="Wants speed.")
    candidate = valid_state(feeling="TODO", thinking="Wants speed.")
    result = StateMonitor().validate(candidate, previous)
    assert any(
        issue.code == "E_CLEARED" and "Feeling" in issue.message
        for issue in result.issues
    )


def test_code_fence_rejected() -> None:
    text = valid_state(current="```\nsecret\n```")
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_STRUCTURE" for issue in result.issues)


def test_control_character_rejected() -> None:
    text = valid_state(current="bad\x00text")
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_CHARS" for issue in result.issues)


def test_sentinel_rejected() -> None:
    text = valid_state(current="<state_edit>oops</state_edit>")
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_SENTINEL" for issue in result.issues)


def test_frame_sentinel_rejected() -> None:
    text = valid_state(current="STATE>>>")
    result = StateMonitor().validate(text)
    assert any(issue.code == "E_SENTINEL" for issue in result.issues)


def test_crlf_is_canonicalized() -> None:
    text = valid_state().replace("\n", "\r\n")
    result = StateMonitor().validate(text)
    assert result.ok
    assert result.chars == len(canonicalize(text))


def test_limit_counts_code_points_not_bytes() -> None:
    goal = "保持采购订单按时交付" * 10
    text = valid_state(goal=goal)
    assert len(text.encode("utf-8")) > len(text)
    result = StateMonitor(4076).validate(text)
    assert result.ok
    assert result.chars == len(canonicalize(text))


def test_extract_state_edit_keeps_action_and_strips_block() -> None:
    edit = valid_state()
    text = (
        'I will expedite.\nACTION: expedite_order(order_id="PO-1234")\n'
        f"<state_edit>\n{edit}</state_edit>"
    )
    parsed = extract_state_edit(text)
    assert parsed.edit == edit.strip()
    assert "ACTION:" in parsed.text
    assert "<state_edit>" not in parsed.text
    assert parsed.issues == ()


def test_extract_multiple_blocks_rejected() -> None:
    parsed = extract_state_edit(
        "<state_edit>a</state_edit><state_edit>b</state_edit>"
    )
    assert parsed.edit is None
    assert any(issue.code == "E_PROTOCOL" for issue in parsed.issues)
    assert "<state_edit>" not in parsed.text


def test_extract_unclosed_block_rejected() -> None:
    parsed = extract_state_edit("answer\n<state_edit>\n## Goal\n...")
    assert parsed.edit is None
    assert any(issue.code == "E_PROTOCOL" for issue in parsed.issues)
    assert parsed.text == "answer"


def test_store_path_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_session_id("../escape")
    with pytest.raises(ValueError):
        StateManager("../escape", StateConfig(root=tmp_path))


def test_corrupt_primary_falls_back_to_backup(config: StateConfig) -> None:
    manager = StateManager("s", config)
    first = manager.ensure("Original goal")
    manager.apply(valid_state(current="Expedited."))
    manager.path.write_text("garbage", encoding="utf-8")

    loaded = manager.load()
    assert loaded.source == "backup"
    assert loaded.content == first
    assert manager.path.read_text(encoding="utf-8") == "garbage"


def test_corrupt_without_backup_is_not_overwritten(config: StateConfig) -> None:
    manager = StateManager("s", config)
    manager.store.write_atomic("s", "garbage")

    recovered = manager.ensure("Original goal")
    assert "Original goal" in recovered
    assert manager.path.read_text(encoding="utf-8") == "garbage"


def test_unreadable_state_is_not_treated_as_missing(config: StateConfig) -> None:
    manager = StateManager("s", config)
    manager.path.mkdir()

    recovered = manager.ensure("Original goal")
    assert "Original goal" in recovered
    assert manager.path.is_dir()


def test_storage_failure_does_not_raise(tmp_path: Path) -> None:
    root = tmp_path / "not_a_directory"
    root.write_text("file", encoding="utf-8")
    manager = StateManager("s", StateConfig(root=root))

    recovered = manager.ensure("Original goal")
    assert "Original goal" in recovered
    assert manager.last_apply is not None
    assert manager.last_apply.status == "rejected"
    assert any(
        issue.code == "E_IO"
        for issue in manager.last_apply.validation.issues
    )


def test_read_other_session_is_read_only(config: StateConfig) -> None:
    a = StateManager("a", config)
    b = StateManager("b", config)
    a.ensure("Goal A")
    b.ensure("Goal B")
    before = b.path.read_text(encoding="utf-8")

    assert a.read_session("b") == before
    assert b.path.read_text(encoding="utf-8") == before
    assert a.read_session("missing") is None


def test_prompt_section_has_budget_and_format(config: StateConfig) -> None:
    manager = StateManager("s", config)
    content = manager.ensure("Original goal")
    section = manager.prompt_section(content)
    assert "Size:" in section
    assert str(config.limit) in section
    assert "## Current State" in section
    assert "never cleared" in section.lower()
    assert section.index("Never clear or reset the state") < section.index(
        "<<<STATE"
    )
    assert section.rstrip().endswith("STATE>>>")


def test_wrapper_appends_state_to_back_of_system_prompt(
    config: StateConfig,
) -> None:
    inner = ScriptedProvider(["plain answer"])
    manager = StateManager("s", config)
    provider = StatefulProvider(inner, manager)

    messages = [
        Message("system", "STATIC SYSTEM PROMPT"),
        Message("user", "Ship PO-1234"),
    ]
    assert provider.complete(messages, []) == "plain answer"

    sent = inner.calls[0]
    assert sent[0].content == "STATIC SYSTEM PROMPT"
    assert sent[1].role == "system"
    assert sent[1].content.startswith("## SESSION STATE PROTOCOL")
    assert sent[1].content.rstrip().endswith("STATE>>>")
    assert sent[2].role == "user"
    assert messages[0].content == "STATIC SYSTEM PROMPT"


def test_wrapper_accepts_valid_edit(config: StateConfig) -> None:
    edit = valid_state(
        goal="Ship PO-1234 on time.",
        current="Expedite approved and executed.",
    )
    response = (
        'Done.\nACTION: expedite_order(order_id="PO-1234")\n'
        f"<state_edit>\n{edit}</state_edit>"
    )
    inner = ScriptedProvider([response])
    manager = StateManager("s", config)
    provider = StatefulProvider(inner, manager)

    result = provider.complete(
        [Message("system", "S"), Message("user", "Ship it")], []
    )
    assert "ACTION:" in result
    assert "<state_edit>" not in result
    assert manager.path.read_text(encoding="utf-8") == edit


def test_wrapper_repairs_rejected_edit(config: StateConfig) -> None:
    bad = "## Goal\nShip it.\n"
    good = valid_state(goal="Ship it.", current="Expedited.")
    original = (
        'ACTION: lookup_order(order_id="PO-1234")\n'
        f"<state_edit>\n{bad}</state_edit>"
    )
    repaired = (
        'ACTION: lookup_order(order_id="PO-1234")\n'
        f"<state_edit>\n{good}</state_edit>"
    )
    inner = ScriptedProvider([original, repaired])
    manager = StateManager("s", config)
    provider = StatefulProvider(inner, manager)

    result = provider.complete(
        [Message("system", "S"), Message("user", "Ship it")], []
    )
    assert len(inner.calls) == 2
    assert "ACTION:" in result
    assert "<state_edit>" not in result
    assert manager.path.read_text(encoding="utf-8") == good


def test_wrapper_discards_after_max_repairs(config: StateConfig) -> None:
    bad = "<state_edit>\n## Goal\nTODO\n</state_edit>"
    responses = [
        f'ACTION: lookup_order(order_id="PO-1234")\n{bad}',
        bad,
        bad,
        bad,
    ]
    inner = ScriptedProvider(responses)
    manager = StateManager(
        "s", StateConfig(root=config.root, max_repairs=3)
    )
    provider = StatefulProvider(inner, manager)

    result = provider.complete(
        [Message("system", "S"), Message("user", "Ship it")], []
    )
    assert len(inner.calls) == 4
    assert "ACTION:" in result
    assert "<state_edit>" not in result
    assert "Ship it" in manager.path.read_text(encoding="utf-8")
    assert "TODO" not in manager.path.read_text(encoding="utf-8")


def test_repair_never_rewrites_visible_answer(config: StateConfig) -> None:
    bad = "## Goal\nShip it.\n"
    good = valid_state(goal="Ship it.", current="Expedited.")
    original = (
        'ACTION: lookup_order(order_id="PO-1234")\n'
        f"<state_edit>\n{bad}</state_edit>"
    )
    repaired = f"A different answer.\n<state_edit>\n{good}</state_edit>"
    inner = ScriptedProvider([original, repaired])
    manager = StateManager("s", config)
    provider = StatefulProvider(inner, manager)

    result = provider.complete(
        [Message("system", "S"), Message("user", "Ship it")], []
    )
    assert "ACTION:" in result
    assert "A different answer" not in result
    assert manager.path.read_text(encoding="utf-8") == good


def test_agent_with_stateful_provider_still_parses_action(
    config: StateConfig,
) -> None:
    edit = valid_state(current="Expedite approved.")
    inner = ScriptedProvider(
        [
            (
                'ACTION: lookup_order(order_id="PO-1234")\n'
                f"<state_edit>\n{edit}</state_edit>"
            ),
            "The order is open.",
        ]
    )
    manager = StateManager("s", config)
    provider = StatefulProvider(inner, manager)

    registry = ToolRegistry()

    def lookup_order(order_id: str) -> str:
        return f"looked up {order_id}"

    registry.add(Tool(lookup_order, "lookup_order", "Look up an order."))
    agent = Agent(provider=provider, registry=registry)

    result = agent.run("Ship it")
    assert result.action == "lookup_order"
    assert "looked up PO-1234" in result.tool_result
    assert "<state_edit>" not in result.proposal
    assert manager.path.read_text(encoding="utf-8") == edit
