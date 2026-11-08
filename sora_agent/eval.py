"""Post-run evaluation.

Evaluation is deliberately **non-executional**: no tool call may depend on a
verdict produced here.  Safety lives in the approval gate and the tool policy,
which fail closed.  A grader error therefore produces a report, not a block.

``structural_grader`` checks what a finished trajectory can prove:

* the final answer is present
* every called tool is in the tool set
* arguments conform to the tool's JSON schema (required fields and types)
* declared ``Tool.requires`` prerequisites succeeded earlier
* a tool never executes after the run stopped
* a tool never executes without the approval it required
* repeated identical calls are reported (score penalty, not a hard failure)

It deliberately does **not** check semantic correctness, whether the answer is
grounded in tool results, the safety or desirability of side effects, business
rules, idempotency, prompt injection, whether a better tool existed, argument
semantics beyond schema shape, or dependencies that a tool never declared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .runner import RunMetrics, Step

NOT_CHECKED = (
    "semantic correctness of the final answer",
    "grounding of claims in tool results",
    "safety or desirability of side effects",
    "business-rule validity",
    "idempotency",
    "prompt injection",
    "whether a better tool or plan existed",
    "dependencies not declared in Tool.requires",
    "argument semantics beyond JSON-schema shape",
)


@dataclass
class Trajectory:
    """Everything a grader may inspect about one finished run."""

    task: str
    steps: list[Step] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    requirements: dict[str, tuple[str, ...]] = field(default_factory=dict)
    final_answer: str = ""
    stop_reason: str = ""
    required_first_tool: str | None = None
    run_id: str = ""


@dataclass
class EvalReport:
    passed: bool
    score: float
    checks: list[str] = field(default_factory=list)
    dimensions: dict[str, float] = field(default_factory=dict)
    error: str = ""
    skipped: bool = False

    def __str__(self) -> str:
        status = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        lines = [f"[{status}] score={self.score:.2f}"]
        if self.error:
            lines.append(f"  ! {self.error}")
        lines.extend(f"  - {check}" for check in self.checks)
        lines.extend(
            f"  * {name}: {value:.2f}"
            for name, value in sorted(self.dimensions.items())
        )
        return "\n".join(lines)


@dataclass
class TrajectoryMetrics:
    """Deterministic measures of the road taken."""

    action_success_rate: float
    redundant_call_rate: float
    steps_to_answer: int
    attempted_tool_calls: int
    successful_tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    provider_seconds: float
    tool_seconds: float


class QualityGate:
    """The evaluation injection point.  Never part of execution."""

    def __init__(
        self, grader: Callable[[Trajectory], EvalReport] | None = None
    ) -> None:
        self.grader = grader or structural_grader

    def evaluate(self, trajectory: Trajectory) -> EvalReport:
        try:
            return self.grader(trajectory)
        except Exception as exc:  # noqa: BLE001 - evaluation must fail soft
            return EvalReport(
                passed=False,
                score=0.0,
                checks=["grader raised an exception"],
                error=f"{type(exc).__name__}: {exc}",
            )


def structural_grader(trajectory: Trajectory) -> EvalReport:
    """Deterministic trajectory checks.  See the module docstring for limits."""
    checks: list[str] = []
    failed = False
    score = 1.0
    seen: set[str] = set()
    prior_success: set[str] = set()
    stopped = False

    if trajectory.stop_reason and trajectory.stop_reason != "completed":
        checks.append(
            f"run stopped with {trajectory.stop_reason!r}, not 'completed'"
        )
        failed = True

    final = trajectory.final_answer.strip()
    if final:
        checks.append("final answer is non-empty")
    else:
        checks.append("final answer is empty")
        failed = True

    results = [result for step in trajectory.steps for result in step.results]
    skipped = [result for result in results if result.skipped]
    if skipped:
        checks.append(
            f"{len(skipped)} tool call(s) were skipped and never executed"
        )
    for result in results:
        name = result.call.name
        schema = trajectory.tools.get(name)
        if schema is None:
            checks.append(f"tool {name!r} is not in the tool set")
            failed = True
        else:
            problems = _schema_errors(schema, result.call.arguments)
            for problem in problems:
                checks.append(f"{name}: {problem}")
            if problems:
                failed = True

        if stopped and result.executed:
            checks.append(f"{name} executed after the run had stopped")
            failed = True
        if result.stopped:
            stopped = True

        if (
            result.approval_required
            and not result.approved
            and result.executed
        ):
            checks.append(f"{name} executed without the required approval")
            failed = True

        for requirement in trajectory.requirements.get(name, ()):
            if requirement not in prior_success:
                checks.append(
                    f"{name} requires {requirement!r} to succeed first"
                )
                failed = True

        if not result.skipped:
            signature = _call_signature(result.call)
            if signature in seen:
                checks.append(
                    f"{name} repeated with identical arguments"
                )
                score -= 0.25
            seen.add(signature)
        if result.executed and not result.is_error:
            prior_success.add(name)

    required = trajectory.required_first_tool
    if required:
        first = next((item for item in results if not item.skipped), None)
        if first is not None and first.call.name != required:
            checks.append(
                f"first action was {first.call.name!r}, expected {required!r}"
            )
            failed = True

    checks.append(
        f"evaluated {len(results)} tool call(s) against "
        f"{len(trajectory.tools)} registered schema(s)"
    )
    return EvalReport(
        passed=not failed,
        score=max(0.0, score) if not failed else 0.0,
        checks=checks,
    )


def trajectory_from_run(
    task: str, run: Any, *, final_answer: str | None = None
) -> Trajectory:
    """Build the evaluation view of a finished run."""
    return Trajectory(
        task=task,
        steps=list(run.steps),
        metrics=run.metrics,
        tools=dict(getattr(run, "tool_schemas", {})),
        requirements=dict(getattr(run, "tool_requirements", {})),
        final_answer=(
            run.final_answer if final_answer is None else final_answer
        ),
        stop_reason=run.stop_reason,
        required_first_tool=getattr(run, "required_first_tool", None),
        run_id=getattr(run, "run_id", ""),
    )


def trajectory_metrics(trajectory: Trajectory) -> TrajectoryMetrics:
    """Derive efficiency, success, and cost measures from a run's metrics."""
    metrics = trajectory.metrics
    attempted = max(
        0, metrics.tool_calls + metrics.skipped_tool_calls
    )
    successful = max(0, metrics.executed_tool_calls - metrics.tool_errors)
    return TrajectoryMetrics(
        action_success_rate=(
            successful / attempted if attempted else 0.0
        ),
        redundant_call_rate=(
            metrics.redundant_tool_calls / attempted if attempted else 0.0
        ),
        steps_to_answer=len(trajectory.steps),
        attempted_tool_calls=attempted,
        successful_tool_calls=successful,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        provider_seconds=metrics.provider_duration,
        tool_seconds=metrics.tool_duration,
    )


def _call_signature(call: Any) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def _schema_errors(
    schema: dict[str, Any], arguments: Any
) -> list[str]:
    if not isinstance(arguments, dict):
        return ["arguments are not an object"]
    parameters = schema.get("parameters")
    if not isinstance(parameters, dict):
        return []
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = parameters.get("required")
    if not isinstance(required, list):
        required = []

    problems: list[str] = []
    for name in required:
        if name not in arguments:
            problems.append(f"missing required argument {name!r}")
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if not isinstance(expected, str):
            continue
        if not _matches_type(value, expected):
            problems.append(
                f"argument {name!r} expected {expected}, got "
                f"{type(value).__name__}"
            )
    return problems


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True
