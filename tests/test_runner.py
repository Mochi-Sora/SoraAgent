from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

from sora_agent.agent import Agent
from sora_agent.approval import ApprovalGate
from sora_agent.identity import IdentityConfig, IdentityStore
from sora_agent.identity_provider import IdentityProvider
from sora_agent.providers import (
    Message,
    ModelTurn,
    OpenAIChatProvider,
    ToolCall,
)
from sora_agent.runner import Runner, RunSpec
from sora_agent.state import StateConfig, StateManager
from sora_agent.state_machine import POStateMachine, State
from sora_agent.state_provider import StatefulProvider
from sora_agent.tools import Tool, ToolExecutor, ToolRegistry


class StructuredProvider:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        return self.turns.pop(0)


class TextProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


class LoopProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        return ModelTurn(
            text="trying again",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.calls}",
                    name="lookup_order",
                    arguments={"order_id": "PO-1"},
                )
            ],
        )


class BrokenProvider:
    def complete_turn(self, messages, tools) -> ModelTurn:
        raise RuntimeError("boom")


class AutoApprove(ApprovalGate):
    def request(self, action: str, detail: str) -> bool:
        return self.approve_for(action, detail)


class RejectAll(ApprovalGate):
    def request(self, action: str, detail: str) -> bool:
        self.log(action, "rejected", detail)
        return False


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.add(
        Tool(
            lambda order_id: f"looked up {order_id}",
            "lookup_order",
            "Look up an order.",
        )
    )
    return tools


def approval_registry() -> ToolRegistry:
    tools = registry()
    tools.add(
        Tool(
            lambda order_id: f"expedited {order_id}",
            "expedite_order",
            "Expedite an order.",
            needs_approval=True,
        )
    )
    return tools


def test_runner_executes_native_tool_call_then_answers() -> None:
    registry_ = registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Looking it up.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            ),
            ModelTurn(text="It is open."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry_)).run(
        RunSpec(
            messages=[Message("system", "S"), Message("user", "Check PO-1")],
            tools=registry_.all(),
            max_steps=5,
        )
    )

    assert run.stop_reason == "completed"
    assert run.final_answer == "It is open."
    assert len(run.steps) == 2
    assert run.steps[0].results[0].executed is True
    assert run.steps[0].results[0].content == "looked up PO-1"
    assert [message.role for message in run.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_message = run.messages[3]
    assert tool_message.name == "lookup_order"
    assert tool_message.tool_call_id == "call_1"


def test_runner_executes_multiple_calls_in_one_turn() -> None:
    registry_ = registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Checking both.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="lookup_order",
                        arguments={"order_id": "PO-2"},
                    ),
                ],
            ),
            ModelTurn(text="Both checked."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry_)).run(
        RunSpec(messages=[Message("user", "Check both")], tools=registry_.all())
    )

    assert run.stop_reason == "completed"
    assert len(run.steps[0].results) == 2
    assert [result.content for result in run.steps[0].results] == [
        "looked up PO-1",
        "looked up PO-2",
    ]
    tool_messages = [
        message for message in run.messages if message.role == "tool"
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "call_1",
        "call_2",
    ]


def test_runner_feeds_unknown_tool_back_as_an_observation() -> None:
    registry_ = registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Trying a tool.",
                tool_calls=[
                    ToolCall(id="call_1", name="missing_tool", arguments={})
                ],
            ),
            ModelTurn(text="I recovered."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry_)).run(
        RunSpec(
            messages=[Message("user", "Do it")],
            tools=registry_.all(),
        )
    )

    assert run.stop_reason == "completed"
    assert run.final_answer == "I recovered."
    assert run.steps[0].results[0].is_error is True
    assert "Unknown tool: missing_tool" in run.steps[0].results[0].content
    assert len(provider.calls) == 2


def test_runner_feeds_invalid_arguments_back_as_an_observation() -> None:
    registry_ = registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Looking it up.",
                tool_calls=[
                    ToolCall(id="call_1", name="lookup_order", arguments={})
                ],
            ),
            ModelTurn(text="Fixed."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry_)).run(
        RunSpec(messages=[Message("user", "Check")], tools=registry_.all())
    )

    assert run.stop_reason == "completed"
    assert run.steps[0].results[0].is_error is True
    assert "Invalid arguments for lookup_order" in run.steps[0].results[0].content
    assert "order_id" in run.steps[0].results[0].content


def test_runner_feeds_tool_exceptions_back_as_observations() -> None:
    tools = ToolRegistry()

    def explode() -> str:
        raise ValueError("downstream exploded")

    tools.add(Tool(explode, "explode", "Always fails."))
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Calling.",
                tool_calls=[ToolCall(id="call_1", name="explode", arguments={})],
            ),
            ModelTurn(text="The tool failed; I will report it."),
        ]
    )

    run = Runner(provider, ToolExecutor(tools)).run(
        RunSpec(messages=[Message("user", "Try")], tools=tools.all())
    )

    assert run.stop_reason == "completed"
    assert run.steps[0].results[0].is_error is True
    assert "tool call failed: downstream exploded" in run.steps[0].results[0].content
    assert len(provider.calls) == 2


def test_runner_approval_rejection_is_terminal() -> None:
    registry_ = approval_registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Expedite it.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            )
        ]
    )
    gate = RejectAll()

    run = Runner(provider, ToolExecutor(registry_, approval=gate)).run(
        RunSpec(messages=[Message("user", "Expedite")], tools=registry_.all())
    )

    assert run.stop_reason == "approval_rejected"
    assert run.final_answer == ""
    assert len(run.steps) == 1
    assert run.steps[0].results[0].executed is False
    assert run.steps[0].results[0].approval_required is True
    assert run.steps[0].results[0].approved is False
    assert run.steps[0].results[0].content == "rejected by human approval gate"
    assert len(provider.calls) == 1


def test_stopped_run_keeps_tool_pairing_for_remaining_calls() -> None:
    registry_ = approval_registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Doing three things.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    ),
                    ToolCall(
                        id="call_3",
                        name="lookup_order",
                        arguments={"order_id": "PO-2"},
                    ),
                ],
            )
        ]
    )

    run = Runner(provider, ToolExecutor(registry_, approval=RejectAll())).run(
        RunSpec(messages=[Message("user", "Go")], tools=registry_.all())
    )

    tool_messages = [
        message for message in run.messages if message.role == "tool"
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "call_1",
        "call_2",
        "call_3",
    ]
    assert tool_messages[1].content == "rejected by human approval gate"
    assert "Skipped: run stopped (approval_rejected)." in tool_messages[2].content


def test_runner_continues_after_approval() -> None:
    registry_ = approval_registry()
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Expedite it.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            ),
            ModelTurn(text="Done."),
        ]
    )
    gate = AutoApprove()

    run = Runner(provider, ToolExecutor(registry_, approval=gate)).run(
        RunSpec(messages=[Message("user", "Expedite")], tools=registry_.all())
    )

    assert run.stop_reason == "completed"
    assert run.steps[0].results[0].approved is True
    assert run.steps[0].results[0].executed is True
    assert any(entry.decision == "approved" for entry in gate.audit)


def test_runner_stops_at_max_steps() -> None:
    registry_ = registry()
    provider = LoopProvider()

    run = Runner(provider, ToolExecutor(registry_)).run(
        RunSpec(
            messages=[Message("user", "Loop")],
            tools=registry_.all(),
            max_steps=2,
        )
    )

    assert run.stop_reason == "max_steps"
    assert run.final_answer == ""
    assert len(run.steps) == 2
    assert provider.calls == 2


def test_runner_truncates_tool_results() -> None:
    tools = ToolRegistry()
    tools.add(
        Tool(
            lambda: "x" * 50,
            "long_result",
            "Return a long result.",
        )
    )
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Calling.",
                tool_calls=[ToolCall(id="call_1", name="long_result", arguments={})],
            ),
            ModelTurn(text="Done."),
        ]
    )

    run = Runner(
        provider,
        ToolExecutor(tools, max_result_chars=10),
    ).run(RunSpec(messages=[Message("user", "Go")], tools=tools.all()))

    content = run.steps[0].results[0].content
    assert content.startswith("x" * 10)
    assert "[truncated 40 chars]" in content


def test_runner_reports_provider_errors() -> None:
    run = Runner(BrokenProvider(), ToolExecutor(registry())).run(
        RunSpec(messages=[Message("user", "Go")], tools=registry().all())
    )

    assert run.stop_reason == "provider_error"
    assert run.error == "RuntimeError: boom"
    assert run.steps == []


def test_agent_adapts_text_actions_at_the_provider_edge() -> None:
    registry_ = registry()
    provider = TextProvider(
        ['ACTION: lookup_order(order_id="PO-9")', "The order is open."]
    )
    agent = Agent(provider=provider, registry=registry_, memory=False, identity=False)

    result = agent.run("Check PO-9")

    assert result.stop_reason == "completed"
    assert result.action == "lookup_order"
    assert "looked up PO-9" in result.tool_result
    assert result.final_answer == "The order is open."
    assert result.proposal == "The order is open."
    assert len(result.steps) == 2
    assert provider.calls[1][-1].role == "tool"


def test_agent_runs_native_calls_and_transitions_state() -> None:
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Expediting.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=approval_registry(),
        approval=AutoApprove(),
        memory=False,
        identity=False,
    )
    state = POStateMachine("PO-1")

    result = agent.run("Expedite PO-1", state=state)

    assert state.state == State.EXPEDITED
    assert result.state == State.EXPEDITED
    assert result.action == "expedite_order"
    assert result.approved is True
    assert result.final_answer == "Done."
    assert result.report is not None and result.report.passed
    assert result.stop_reason == "completed"


def test_rejected_approval_does_not_transition_state() -> None:
    provider = StructuredProvider(
        [
            ModelTurn(
                text="Expediting.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            )
        ]
    )
    agent = Agent(
        provider=provider,
        registry=approval_registry(),
        approval=RejectAll(),
        memory=False,
        identity=False,
    )
    state = POStateMachine("PO-1")

    result = agent.run("Expedite PO-1", state=state)

    assert result.stop_reason == "approval_rejected"
    assert result.approved is False
    assert state.state == State.OPEN
    assert "rejected by human approval gate" in result.tool_result


def test_identity_wrapper_preserves_native_tool_calls(
    tmp_path: Path,
) -> None:
    store = IdentityStore(IdentityConfig(root=tmp_path / "identity"))

    class Provider:
        def complete_turn(self, messages, tools) -> ModelTurn:
            return ModelTurn(
                text="Noted.\n<user_edit>\n- Name: Jaden\n</user_edit>",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            )

        def complete(self, messages, tools) -> str:
            return "unused"

    turn = IdentityProvider(Provider(), store).complete_turn(
        [Message("user", "Remember me")], []
    )

    assert turn.text == "Noted."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "lookup_order"
    assert store.load("user").content == "- Name: Jaden\n"


def test_state_wrapper_preserves_native_tool_calls(
    tmp_path: Path,
) -> None:
    manager = StateManager("s", StateConfig(root=tmp_path / "state"))

    class Provider:
        def complete_turn(self, messages, tools) -> ModelTurn:
            return ModelTurn(
                text="Looking it up.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    )
                ],
            )

        def complete(self, messages, tools) -> str:
            return "unused"

    turn = StatefulProvider(Provider(), manager).complete_turn(
        [Message("user", "Check PO-1")], []
    )

    assert turn.text == "Looking it up."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "lookup_order"
    assert manager.path.exists()


def test_openai_provider_parses_native_tool_calls(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeCompletions:
        def create(self, **payload):
            captured.update(payload)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="lookup_order",
                                        arguments='{"order_id": "PO-1"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient)
    )
    tools = registry()
    provider = OpenAIChatProvider(model="test", api_key="test")

    turn = provider.complete_turn([Message("user", "hi")], tools.all())

    assert turn.text == ""
    assert turn.tool_calls[0].name == "lookup_order"
    assert turn.tool_calls[0].arguments == {"order_id": "PO-1"}
    assert captured["tools"][0]["function"]["name"] == "lookup_order"  # type: ignore[index]

    provider.complete_turn(
        [
            Message("user", "hi"),
            Message("assistant", "", tool_calls=turn.tool_calls),
            Message(
                "tool",
                "looked up PO-1",
                tool_call_id="call_1",
                name="lookup_order",
            ),
        ],
        [],
    )
    sent = captured["messages"]  # type: ignore[index]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"order_id": "PO-1"}'
    assert sent[2]["tool_call_id"] == "call_1"
    assert sent[2]["name"] == "lookup_order"


def test_openai_provider_reports_malformed_arguments(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **payload):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Calling.",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="lookup_order",
                                        arguments="{not json",
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient)
    )
    provider = OpenAIChatProvider(model="test", api_key="test")

    turn = provider.complete_turn([Message("user", "hi")], registry().all())

    assert turn.tool_calls[0].arguments == {}
    assert "invalid JSON arguments" in turn.tool_calls[0].error


@pytest.mark.parametrize("max_steps", [0, -1])
def test_agent_zero_step_budget_does_not_call_provider(max_steps: int) -> None:
    provider = TextProvider(["never"])
    agent = Agent(
        provider=provider,
        registry=registry(),
        memory=False,
        identity=False,
        max_steps=max_steps,
    )

    result = agent.run("No-op")

    assert result.stop_reason == "max_steps"
    assert provider.calls == []
