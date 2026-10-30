"""Tests for lazy tool discovery and the required first action."""

from __future__ import annotations

from typing import Any

import pytest

from sora_agent.agent import Agent
from sora_agent.approval import ApprovalGate
from sora_agent.eval import EvalReport, QualityGate
from sora_agent.memory.gate import RecallEnforcingProvider
from sora_agent.providers import Message, MockProvider, ModelTurn, ToolCall
from sora_agent.runner import Runner, RunSpec
from sora_agent.state_machine import POStateMachine, State
from sora_agent.tool_search import (
    FIND_TOOLS_NAME,
    MATCH_LIMIT,
    MIN_SCORE,
    FindToolsTool,
    ToolGate,
    score_need,
)
from sora_agent.tools import Tool, ToolExecutor, ToolRegistry


class FakeMemoryTool:
    name = "memory_recall"
    description = "Read relevant memory as untrusted evidence."
    needs_approval = False

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "string"},
                    "k": {"type": "integer"},
                },
                "required": ["query"],
            },
        }

    def call(self, query: str, scope: str = "auto", k: int = 8) -> str:
        return f"memory for {query} ({scope}, {k})"


class FakeMemory:
    def __init__(self) -> None:
        self.tool_calls = 0
        self.recall_tool = FakeMemoryTool()

    def tool(self) -> FakeMemoryTool:
        self.tool_calls += 1
        return self.recall_tool

    def before_run(self, task: str, state: Any = None) -> None:
        self.task = task

    def after_run(self, result: Any, state: Any = None) -> None:
        self.result = result


class RecordingProvider:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []
        self.tools: list[list[str]] = []
        self.schemas: list[list[dict[str, Any]]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        self.tools.append([getattr(tool, "name", "") for tool in tools])
        self.schemas.append(
            [getattr(tool, "schema", {}) for tool in tools]
        )
        return self.turns.pop(0)


class AutoApprove(ApprovalGate):
    def request(self, action: str, detail: str) -> bool:
        return self.approve_for(action, detail)


def order_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(
        Tool(
            lambda order_id: f"looked up {order_id}",
            "lookup_order",
            "Return status and details for a purchase order id.",
        )
    )
    registry.add(
        Tool(
            lambda order_id: f"expedited {order_id}",
            "expedite_order",
            "Mark an order as expedited (needs human approval).",
            needs_approval=True,
        )
    )
    registry.add(
        Tool(
            lambda order_id: f"cancelled {order_id}",
            "cancel_order",
            "Cancel a purchase order (needs human approval).",
            needs_approval=True,
        )
    )
    return registry


def discovery_gate(registry: ToolRegistry) -> tuple[ToolGate, FindToolsTool]:
    gate = ToolGate(registry, always_visible={FIND_TOOLS_NAME})
    find = FindToolsTool(gate)
    registry.add(find)
    return gate, find


def test_registry_copy_is_independent() -> None:
    original = order_registry()
    clone = original.copy()
    clone.add(Tool(lambda: "x", "extra", "Extra tool."))

    assert original.has("lookup_order")
    assert not original.has("extra")
    assert clone.has("extra")


def test_gate_hides_everything_except_always_visible() -> None:
    registry = order_registry()
    gate, _find = discovery_gate(registry)

    assert gate.visible_names() == [FIND_TOOLS_NAME]
    assert {tool.name for tool in gate.hidden()} == {
        "lookup_order",
        "expedite_order",
        "cancel_order",
    }

    added = gate.unlock(["expedite_order", "missing_tool"])

    assert added == ["expedite_order"]
    assert gate.visible_names() == [FIND_TOOLS_NAME, "expedite_order"]
    assert gate.is_visible("expedite_order")
    assert not gate.is_visible("lookup_order")
    assert gate.unlock(["expedite_order"]) == []


def test_catalog_shrinks_after_unlock() -> None:
    registry = order_registry()
    gate, _find = discovery_gate(registry)

    before = gate.catalog()
    assert "`lookup_order`" in before
    assert "`expedite_order`" in before

    gate.unlock(["expedite_order"])
    after = gate.catalog()

    assert "`expedite_order`" not in after
    assert "`lookup_order`" in after


def test_catalog_is_capped() -> None:
    registry = ToolRegistry()
    for index in range(65):
        registry.add(
            Tool(lambda: "", f"tool_{index:02d}", f"Tool number {index}.")
        )
    gate = ToolGate(registry)
    catalog = gate.catalog()

    assert catalog.count("- `tool_") == 60
    assert "(5 more not listed" in catalog


def test_ranking_prefers_name_over_description_noise() -> None:
    exact = score_need(
        "expedite the order",
        "expedite_order",
        "Mark an order as expedited.",
    )
    noise = score_need(
        "expedite the order",
        "lookup_order",
        "Return status and details for a purchase order id.",
    )

    assert exact > noise * 1.5
    assert score_need("send a fax", "lookup_order", "Look up an order.") < MIN_SCORE


def test_ranking_matches_description_words_and_typos() -> None:
    description = score_need(
        "cancel a purchase",
        "lookup_order",
        "Cancel a purchase order after approval.",
    )
    typo = score_need(
        "expidite order",
        "expedite_order",
        "Mark an order as expedited.",
    )

    assert description >= MIN_SCORE
    assert typo >= MIN_SCORE


def test_find_tools_loads_matches_with_signatures() -> None:
    registry = order_registry()
    gate, find = discovery_gate(registry)

    result = find.call("expedite a delayed purchase order")

    assert "Tools for 'expedite a delayed purchase order':" in result
    assert "`expedite_order` (now loaded" in result
    assert "expedite_order(order_id: string)" in result
    assert "Loaded tools appear in your tool list" in result
    assert "expedite_order" in gate.visible_names()


def test_find_tools_no_match_lists_available_tools() -> None:
    registry = order_registry()
    _gate, find = discovery_gate(registry)

    result = find.call("send a fax to accounting")

    assert "No tool matched" in result
    assert "Available tools:" in result
    assert "`lookup_order`" in result
    assert "Try different words" in result


def test_find_tools_rejects_empty_need() -> None:
    registry = order_registry()
    _gate, find = discovery_gate(registry)

    with pytest.raises(ValueError, match="non-empty need"):
        find.call("   ")

    result = ToolExecutor(registry).execute(
        ToolCall(id="call_1", name=FIND_TOOLS_NAME, arguments={"need": ""})
    )
    assert result.is_error is True
    assert "non-empty need" in result.content


def test_catalog_lives_only_in_find_tools_schema() -> None:
    registry = order_registry()
    gate, find = discovery_gate(registry)
    visible = gate.visible()

    schemas = {tool.name: tool.schema for tool in visible}
    assert "`lookup_order`" in schemas[FIND_TOOLS_NAME]["description"]
    assert "Not loaded yet" in schemas[FIND_TOOLS_NAME]["description"]
    for name, schema in schemas.items():
        if name != FIND_TOOLS_NAME:
            assert "Not loaded yet" not in schema["description"]


def test_agent_exposes_only_find_tools_and_memory_recall_first() -> None:
    memory = FakeMemory()
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "expedite a late purchase order"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="expedite_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        approval=AutoApprove(),
        memory=memory,
        identity=False,
    )

    result = agent.run("Expedite PO-1.", state=POStateMachine("PO-1"))

    assert provider.tools[0] == ["memory_recall", FIND_TOOLS_NAME]
    assert FIND_TOOLS_NAME in provider.tools[1]
    assert "expedite_order" in provider.tools[2]
    assert [step.tools for step in result.steps] == provider.tools
    assert result.stop_reason == "completed"
    assert result.final_answer == "Done."
    assert result.steps[0].results[0].executed is True
    assert result.steps[0].results[0].call.name == "memory_recall"
    # Memory content arrives only through the tool result, never preloaded.
    assert not any(
        "memory for PO-1" in message.content
        for message in provider.calls[0]
    )
    assert provider.calls[1][-1].role == "tool"
    assert "memory for PO-1" in provider.calls[1][-1].content


def test_agent_catalog_shrinks_in_the_next_schema() -> None:
    memory = FakeMemory()
    registry = order_registry()
    registry.add(
        Tool(
            lambda to, subject: f"sent to {to}: {subject}",
            "send_email",
            "Send an email message to a recipient.",
        )
    )
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "expedite a late purchase order"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        memory=memory,
        identity=False,
    )

    agent.run("Expedite PO-1.")

    first_find = next(
        schema
        for schema in provider.schemas[0]
        if schema["name"] == FIND_TOOLS_NAME
    )
    next_find = next(
        schema
        for schema in provider.schemas[2]
        if schema["name"] == FIND_TOOLS_NAME
    )
    assert "`expedite_order`" in first_find["description"]
    assert "`memory_recall`" not in first_find["description"]
    assert "`find_tools`" not in first_find["description"]
    assert "`expedite_order`" not in next_find["description"]
    assert "`send_email`" in next_find["description"]


def test_agent_prompt_mentions_discovery_and_recall() -> None:
    memory = FakeMemory()
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "anything"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        memory=memory,
        identity=False,
    )

    agent.run("Check the order.")

    prompt = provider.calls[0][0].content
    assert "find_tools" in prompt
    assert "memory_recall" in prompt
    assert "first action" in prompt


def test_agent_without_memory_has_no_recall_requirement_or_prompt() -> None:
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "look up an order"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="lookup_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        memory=False,
        identity=False,
    )

    result = agent.run("Check PO-1.")

    assert provider.tools[0] == [FIND_TOOLS_NAME]
    assert "memory_recall" not in provider.calls[0][0].content
    assert "`lookup_order` (now loaded" in result.steps[0].results[0].content
    assert result.action == "lookup_order"
    assert result.stop_reason == "completed"


def test_agent_does_not_wrap_provider_with_forced_recall() -> None:
    provider = RecordingProvider([ModelTurn(text="Done.")])
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        memory=FakeMemory(),
        identity=False,
    )

    assert not isinstance(agent.provider, RecallEnforcingProvider)


def test_agent_rejects_direct_answer_and_stops_after_two_corrections() -> None:
    provider = RecordingProvider(
        [
            ModelTurn(text="I can answer without memory."),
            ModelTurn(text="Really, no memory needed."),
            ModelTurn(text="Still answering directly."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        memory=FakeMemory(),
        identity=False,
        max_steps=5,
    )

    result = agent.run("Check PO-1.")

    assert result.stop_reason == "memory_recall_required"
    assert result.final_answer == ""
    assert len(result.steps) == 3
    assert len(provider.calls) == 3
    for call in provider.calls[1:]:
        assert call[-1].role == "user"
        assert "first action must be `memory_recall`" in call[-1].content
    assert all(step.tools[:1] == ["memory_recall"] for step in result.steps)


def test_agent_skips_wrong_first_tool_then_retries_recall() -> None:
    called: list[str] = []
    registry = order_registry()
    registry.add(
        Tool(
            lambda order_id: called.append(order_id) or "looked up",
            "guarded_lookup",
            "A tool that must not run before recall.",
        )
    )
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="wrong_1",
                        name="guarded_lookup",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        memory=FakeMemory(),
        identity=False,
    )

    result = agent.run("Check PO-1.")

    assert called == []
    assert result.stop_reason == "completed"
    assert result.steps[0].results[0].executed is False
    assert result.steps[0].results[0].skipped is True
    assert "first action must be `memory_recall`" in (
        result.steps[0].results[0].content
    )
    assert result.steps[1].results[0].executed is True
    assert result.action == "memory_recall"


def test_agent_executes_recall_and_skips_other_calls_in_the_same_turn() -> None:
    called: list[str] = []
    registry = order_registry()
    registry.add(
        Tool(
            lambda order_id: called.append(order_id) or "looked up",
            "guarded_lookup",
            "A tool that must wait for recall.",
        )
    )
    provider = RecordingProvider(
        [
            ModelTurn(
                text="Thinking about the order.",
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    ),
                    ToolCall(
                        id="wrong_1",
                        name="guarded_lookup",
                        arguments={"order_id": "PO-1"},
                    ),
                ],
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        memory=FakeMemory(),
        identity=False,
    )

    result = agent.run("Check PO-1.")

    assert called == []
    assert result.stop_reason == "completed"
    assert [item.call.name for item in result.steps[0].results] == [
        "memory_recall",
        "guarded_lookup",
    ]
    assert result.steps[0].results[0].executed is True
    assert result.steps[0].results[1].executed is False
    assert result.steps[0].results[1].skipped is True
    assert "must complete before other tools" in (
        result.steps[0].results[1].content
    )
    assert result.action == "memory_recall"


def test_recall_prelude_bypasses_the_quality_guard() -> None:
    guarded: list[str] = []

    def guard(proposal: str, _context: str):
        guarded.append(proposal)
        return True, None

    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        memory=FakeMemory(),
        identity=False,
    )

    # Use the runner directly so the gate's private guard can be observed.
    run_registry = agent.registry.copy()
    run_registry.add(agent.memory_tool)
    run = Runner(provider, ToolExecutor(run_registry)).run(
        RunSpec(
            messages=[Message("user", "Check PO-1.")],
            tools=run_registry.all(),
            required_first_tool="memory_recall",
            guard=guard,
        )
    )

    assert run.stop_reason == "completed"
    assert guarded == []


def test_runner_requires_first_tool_before_a_direct_answer() -> None:
    registry = ToolRegistry()
    registry.add(Tool(lambda: "memory", "memory_recall", "Recall."))
    provider = RecordingProvider(
        [
            ModelTurn(text="No tools needed."),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry)).run(
        RunSpec(
            messages=[Message("user", "Go")],
            tools=registry.all(),
            required_first_tool="memory_recall",
        )
    )

    assert run.stop_reason == "completed"
    assert len(run.steps) == 3
    assert run.messages[1].role == "assistant"
    assert run.messages[2].role == "user"
    assert "first action must be" in run.messages[2].content


def test_runner_hard_stops_after_two_failed_corrections() -> None:
    registry = ToolRegistry()
    registry.add(Tool(lambda: "memory", "memory_recall", "Recall."))
    provider = RecordingProvider(
        [ModelTurn(text="one"), ModelTurn(text="two"), ModelTurn(text="three")]
    )

    run = Runner(provider, ToolExecutor(registry)).run(
        RunSpec(
            messages=[Message("user", "Go")],
            tools=registry.all(),
            required_first_tool="memory_recall",
        )
    )

    assert run.stop_reason == "memory_recall_required"
    assert len(run.steps) == 3
    assert run.final_answer == ""


def test_runner_skips_extra_calls_after_required_recall() -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.add(Tool(lambda: "memory", "memory_recall", "Recall."))
    registry.add(
        Tool(
            lambda: executed.append("business") or "done",
            "business",
            "Do business.",
        )
    )
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(id="recall_1", name="memory_recall", arguments={}),
                    ToolCall(id="biz_1", name="business", arguments={}),
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry)).run(
        RunSpec(
            messages=[Message("user", "Go")],
            tools=registry.all(),
            required_first_tool="memory_recall",
        )
    )

    assert executed == []
    assert run.stop_reason == "completed"
    assert [item.call.name for item in run.steps[0].results] == [
        "memory_recall",
        "business",
    ]
    assert run.steps[0].results[1].skipped is True
    assert "must complete before other tools" in (
        run.steps[0].results[1].content
    )


def test_step_records_the_visible_tool_set_from_the_provider() -> None:
    registry = order_registry()
    gate, _find = discovery_gate(registry)
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "look up an order"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
        ]
    )

    run = Runner(provider, ToolExecutor(registry)).run(
        RunSpec(
            messages=[Message("user", "Go")],
            tools=gate.visible(),
            tool_provider=gate.visible,
        )
    )

    assert run.steps[0].tools == [FIND_TOOLS_NAME]
    assert "lookup_order" in run.steps[1].tools
    assert len(run.steps[0].tools) < len(run.steps[1].tools)


def test_discovery_resets_between_runs() -> None:
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "look up an order"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="lookup_1",
                        name="lookup_order",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            ),
            ModelTurn(text="Done."),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_2",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "look up an order"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="lookup_2",
                        name="lookup_order",
                        arguments={"order_id": "PO-2"},
                    )
                ]
            ),
            ModelTurn(text="Done again."),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        memory=False,
        identity=False,
    )

    agent.run("Check PO-1.")
    agent.run("Check PO-2.")

    assert provider.tools[0] == [FIND_TOOLS_NAME]
    assert provider.tools[3] == [FIND_TOOLS_NAME]
    assert "lookup_order" in provider.tools[4]


def test_reserved_find_tools_name_is_rejected() -> None:
    registry = order_registry()
    registry.add(Tool(lambda: "mine", FIND_TOOLS_NAME, "User tool."))

    with pytest.raises(ValueError, match="reserved"):
        Agent(provider=RecordingProvider([]), registry=registry, memory=False)


def test_discovery_flow_still_transitions_state_after_approval() -> None:
    memory = FakeMemory()
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "expedite a late purchase order"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="expedite_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            ),
            ModelTurn(text="Expedited."),
        ]
    )
    approval = AutoApprove()
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        approval=approval,
        memory=memory,
        identity=False,
    )
    state = POStateMachine("PO-1")

    result = agent.run("Expedite PO-1.", state=state)

    assert result.stop_reason == "completed"
    assert result.action == "expedite_order"
    assert result.approved is True
    assert state.state == State.EXPEDITED
    assert any(entry.decision == "approved" for entry in approval.audit)


def test_search_unlocks_at_most_the_match_limit() -> None:
    registry = ToolRegistry()
    for index in range(10):
        registry.add(
            Tool(
                lambda: "",
                f"order_tool_{index:02d}",
                "Manage a purchase order.",
            )
        )
    gate, find = discovery_gate(registry)

    find.call("manage a purchase order")

    assert len(gate.visible_names()) == MATCH_LIMIT + 1


def test_quality_gate_still_blocks_business_actions_after_recall() -> None:
    quality = QualityGate(
        lambda proposal, context: EvalReport(
            "expedite_order(" not in proposal,
            1.0 if "expedite_order(" not in proposal else 0.0,
            ["business action checked"],
        )
    )
    provider = RecordingProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="recall_1",
                        name="memory_recall",
                        arguments={"query": "PO-1"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "expedite a late purchase order"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="expedite_1",
                        name="expedite_order",
                        arguments={"order_id": "PO-1"},
                    )
                ]
            ),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        approval=AutoApprove(),
        quality=quality,
        memory=FakeMemory(),
        identity=False,
    )
    state = POStateMachine("PO-1")

    result = agent.run("Expedite PO-1.", state=state)

    assert result.stop_reason == "quality_blocked"
    assert state.state == State.OPEN
    assert result.steps[-1].results[0].executed is False


def test_mock_provider_follows_recall_discovery_and_action() -> None:
    provider = MockProvider()
    agent = Agent(
        provider=provider,
        registry=order_registry(),
        approval=AutoApprove(),
        memory=FakeMemory(),
        identity=False,
    )
    state = POStateMachine("PO-1234")

    result = agent.run(
        "Purchase order PO-1234 is stuck; the supplier can't ship on time.",
        state=state,
    )

    assert result.stop_reason == "completed"
    assert result.action == "expedite_order"
    assert result.approved is True
    assert state.state == State.EXPEDITED
    assert [step.results[0].call.name for step in result.steps if step.results] == [
        "memory_recall",
        FIND_TOOLS_NAME,
        "expedite_order",
    ]


def test_mock_provider_discovers_without_memory() -> None:
    agent = Agent(
        provider=MockProvider(),
        registry=order_registry(),
        memory=False,
        identity=False,
    )

    result = agent.run("Check PO-1234.")

    assert result.stop_reason == "completed"
    assert result.action == "lookup_order"
    assert result.steps[0].tools == [FIND_TOOLS_NAME]
    assert [step.results[0].call.name for step in result.steps if step.results] == [
        FIND_TOOLS_NAME,
        "lookup_order",
    ]
