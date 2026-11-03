"""Tests for the non-executional quality gate and trajectory metrics."""

from __future__ import annotations

from sora_agent.agent import Agent
from sora_agent.eval import (
    EvalReport,
    QualityGate,
    Trajectory,
    trajectory_metrics,
)
from sora_agent.providers import ModelTurn, ToolCall
from sora_agent.runner import RunMetrics, Step
from sora_agent.tools import Tool, ToolRegistry, ToolResult

LOOKUP_SCHEMA = {
    "name": "lookup_order",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}
EXPEDITE_SCHEMA = {
    "name": "expedite_order",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}
TOOLS = {"lookup_order": LOOKUP_SCHEMA, "expedite_order": EXPEDITE_SCHEMA}


def _result(
    name: str = "lookup_order",
    arguments: dict | None = None,
    *,
    executed: bool = True,
    is_error: bool = False,
    approval_required: bool = False,
    approved: bool = False,
    skipped: bool = False,
    stopped: bool = False,
    stop_reason: str = "",
) -> ToolResult:
    return ToolResult(
        call=ToolCall(
            "c1",
            name,
            {"order_id": "PO-1"} if arguments is None else arguments,
        ),
        content="ok",
        is_error=is_error,
        approval_required=approval_required,
        approved=approved,
        executed=executed,
        skipped=skipped,
        stopped=stopped,
        stop_reason=stop_reason,
    )


def _trajectory(
    steps: list[Step],
    *,
    answer: str = "done",
    tools: dict | None = None,
    requirements: dict | None = None,
    required: str | None = None,
    metrics: RunMetrics | None = None,
) -> Trajectory:
    return Trajectory(
        task="task",
        steps=steps,
        metrics=metrics or RunMetrics(),
        tools=TOOLS if tools is None else tools,
        requirements=requirements or {},
        final_answer=answer,
        stop_reason="completed",
        required_first_tool=required,
    )


def _step(index: int, results: list[ToolResult]) -> Step:
    return Step(index=index, turn=ModelTurn(), results=results)


def test_a_clean_trajectory_passes():
    report = QualityGate().evaluate(
        _trajectory([_step(0, [_result()]), _step(1, [])])
    )
    assert report.passed is True
    assert report.score == 1.0
    assert report.error == ""
    assert "final answer is non-empty" in report.checks


def test_empty_answer_fails():
    report = QualityGate().evaluate(_trajectory([], answer=""))
    assert report.passed is False
    assert any("empty" in check for check in report.checks)


def test_unknown_tool_fails():
    report = QualityGate().evaluate(
        _trajectory([_step(0, [_result("not_registered")])])
    )
    assert report.passed is False
    assert any("not in the tool set" in check for check in report.checks)


def test_schema_type_mismatch_fails():
    report = QualityGate().evaluate(
        _trajectory([_step(0, [_result(arguments={"order_id": 5})])])
    )
    assert report.passed is False
    assert any("expected string" in check for check in report.checks)


def test_missing_required_argument_fails():
    report = QualityGate().evaluate(
        _trajectory([_step(0, [_result(arguments={})])])
    )
    assert report.passed is False
    assert any("missing required argument" in check for check in report.checks)


def test_missing_prerequisite_fails():
    requirements = {"expedite_order": ("lookup_order",)}
    report = QualityGate().evaluate(
        _trajectory(
            [_step(0, [_result("expedite_order")])],
            requirements=requirements,
        )
    )
    assert report.passed is False
    assert any("requires 'lookup_order'" in check for check in report.checks)


def test_satisfied_prerequisite_passes():
    requirements = {"expedite_order": ("lookup_order",)}
    report = QualityGate().evaluate(
        _trajectory(
            [
                _step(0, [_result("lookup_order")]),
                _step(1, [_result("expedite_order")]),
            ],
            requirements=requirements,
        )
    )
    assert report.passed is True


def test_failed_prerequisite_does_not_count():
    requirements = {"expedite_order": ("lookup_order",)}
    report = QualityGate().evaluate(
        _trajectory(
            [
                _step(0, [_result("lookup_order", is_error=True)]),
                _step(1, [_result("expedite_order")]),
            ],
            requirements=requirements,
        )
    )
    assert report.passed is False


def test_duplicate_calls_penalize_but_do_not_fail():
    report = QualityGate().evaluate(
        _trajectory(
            [
                _step(0, [_result("lookup_order")]),
                _step(1, [_result("lookup_order")]),
            ]
        )
    )
    assert report.passed is True
    assert report.score == 0.75
    assert any("repeated" in check for check in report.checks)


def test_unapproved_execution_fails():
    report = QualityGate().evaluate(
        _trajectory(
            [
                _step(
                    0,
                    [
                        _result(
                            "expedite_order",
                            approval_required=True,
                            approved=False,
                            executed=True,
                        )
                    ],
                )
            ]
        )
    )
    assert report.passed is False
    assert any("without the required approval" in check for check in report.checks)


def test_execution_after_stop_fails():
    report = QualityGate().evaluate(
        _trajectory(
            [
                _step(
                    0,
                    [
                        _result(
                            "expedite_order",
                            executed=False,
                            stopped=True,
                            stop_reason="approval_rejected",
                        ),
                        _result("lookup_order", executed=True),
                    ],
                )
            ]
        )
    )
    assert report.passed is False
    assert any("after the run had stopped" in check for check in report.checks)


def test_required_first_tool_violation_fails():
    report = QualityGate().evaluate(
        _trajectory(
            [_step(0, [_result("lookup_order")])],
            required="memory_recall",
        )
    )
    assert report.passed is False
    assert any("first action" in check for check in report.checks)


def test_a_grader_exception_becomes_a_report():
    def broken(trajectory):
        raise RuntimeError("grader is down")

    report = QualityGate(broken).evaluate(_trajectory([]))
    assert report.passed is False
    assert "grader is down" in report.error


def test_trajectory_metrics_define_asr_and_redundancy():
    metrics = RunMetrics(
        tool_calls=4,
        executed_tool_calls=3,
        tool_errors=1,
        redundant_tool_calls=1,
    )
    efficiency = trajectory_metrics(
        _trajectory(
            [_step(0, []), _step(1, []), _step(2, [])], metrics=metrics
        )
    )

    assert efficiency.attempted_tool_calls == 4
    assert efficiency.successful_tool_calls == 2
    assert efficiency.action_success_rate == 0.5
    assert efficiency.redundant_call_rate == 0.25
    assert efficiency.steps_to_answer == 3


def test_a_real_run_with_a_failing_gate_still_executes():
    executed: list[str] = []

    class Provider:
        def __init__(self) -> None:
            self.turns = [
                ModelTurn(
                    tool_calls=[
                        ToolCall(
                            "c1",
                            "lookup_order",
                            {"order_id": "PO-1"},
                        )
                    ]
                ),
                ModelTurn(text="done"),
            ]

        def complete_turn(self, messages, tools) -> ModelTurn:
            return self.turns.pop(0)

    registry = ToolRegistry()
    registry.add(
        Tool(
            lambda order_id: executed.append(order_id) or "open",
            "lookup_order",
            "Look up an order.",
        )
    )
    gate = QualityGate(lambda trajectory: EvalReport(False, 0.0, ["no"]))
    agent = Agent(
        provider=Provider(),
        registry=registry,
        quality=gate,
        memory=False,
        identity=False,
    )

    result = agent.run("Look up PO-1.")

    assert result.stop_reason == "completed"
    assert executed == ["PO-1"]
    assert result.report is not None
    assert result.report.passed is False
    assert result.action == "lookup_order"
