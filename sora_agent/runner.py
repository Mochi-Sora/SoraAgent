"""The pure model/tool loop.

``Runner`` knows nothing about sessions, files, identity, state, or the CLI.  It
receives messages and tools, asks the provider for a turn, hands tool calls to an
executor, appends the observations, and repeats until the model answers, a policy
stops the run, or the step budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .providers import Message, ModelTurn, ToolCall, provider_turn
from .tools import ToolExecutor, ToolResult


@dataclass
class Step:
    """One assistant turn and the tool results it produced."""

    index: int
    turn: ModelTurn
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class RunSpec:
    messages: list[Message]
    tools: list[Any]
    max_steps: int = 10
    context: str = ""
    # Called once per action-bearing turn, before any tool runs. Returns
    # (passed, report). A failed guard stops the run with quality_blocked.
    guard: Callable[[str, str], tuple[bool, Any]] | None = None


@dataclass
class RunResult:
    messages: list[Message]
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = "max_steps"
    report: Any = None
    error: str = ""

    @property
    def tool_results(self) -> list[ToolResult]:
        return [result for step in self.steps for result in step.results]


class Runner:
    """Run one task through the provider/tool loop."""

    def __init__(self, provider: Any, executor: ToolExecutor) -> None:
        self.provider = provider
        self.executor = executor

    def run(self, spec: RunSpec) -> RunResult:
        messages = list(spec.messages)
        steps: list[Step] = []
        final_answer = ""
        stop_reason = "max_steps"
        report: Any = None
        error = ""

        for index in range(max(0, int(spec.max_steps))):
            try:
                turn = provider_turn(self.provider, messages, spec.tools)
            except Exception as exc:  # noqa: BLE001 - surface provider failures
                stop_reason = "provider_error"
                error = f"{type(exc).__name__}: {exc}"
                break

            if not turn.tool_calls:
                messages.append(Message("assistant", turn.text))
                steps.append(Step(index=index, turn=turn))
                final_answer = turn.text
                stop_reason = "completed"
                break

            messages.append(
                Message(
                    role="assistant",
                    content=turn.text,
                    tool_calls=list(turn.tool_calls),
                )
            )

            if spec.guard is not None:
                proposal = "\n".join(
                    part
                    for part in (
                        turn.text.strip(),
                        _render_calls(turn.tool_calls),
                    )
                    if part
                )
                try:
                    passed, guard_report = spec.guard(
                        proposal, spec.context
                    )
                except Exception as exc:  # noqa: BLE001 - a broken gate blocks
                    passed, guard_report = False, _GuardFailure(str(exc))
                if report is None and guard_report is not None:
                    report = guard_report
                if not passed:
                    results = _blocked_results(
                        turn.tool_calls, guard_report
                    )
                    for result in results:
                        messages.append(_tool_message(result))
                    steps.append(
                        Step(index=index, turn=turn, results=results)
                    )
                    stop_reason = "quality_blocked"
                    break

            results: list[ToolResult] = []
            stopped = ""
            for call in turn.tool_calls:
                try:
                    result = self.executor.execute(call)
                except Exception as exc:  # noqa: BLE001 - never lose a tool message
                    result = ToolResult(
                        call=call,
                        content=f"tool executor failed: {exc}",
                        is_error=True,
                        stopped=True,
                        stop_reason="tool_error",
                    )
                results.append(result)
                messages.append(_tool_message(result))
                if report is None and result.report is not None:
                    report = result.report
                if result.stopped:
                    stopped = result.stop_reason or "stopped"
                    break

            if stopped:
                # Keep the assistant/tool pairing valid even though the run
                # ends before the remaining calls execute.
                for call in turn.tool_calls[len(results):]:
                    messages.append(
                        Message(
                            role="tool",
                            content=f"Skipped: run stopped ({stopped}).",
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
            steps.append(Step(index=index, turn=turn, results=results))
            if stopped:
                stop_reason = stopped
                break

        return RunResult(
            messages=messages,
            steps=steps,
            final_answer=final_answer,
            stop_reason=stop_reason,
            report=report,
            error=error,
        )


class _GuardFailure:
    def __init__(self, message: str) -> None:
        self.passed = False
        self.message = message

    def __str__(self) -> str:
        return f"[FAIL] guard failed: {self.message}"


def _blocked_results(
    calls: list[ToolCall], report: Any
) -> list[ToolResult]:
    content = "Blocked by quality gate"
    if report is not None:
        content += ":\n" + str(report)
    return [
        ToolResult(
            call=call,
            content=content,
            is_error=True,
            stopped=True,
            stop_reason="quality_blocked",
            report=report,
        )
        for call in calls
    ]


def _tool_message(result: ToolResult) -> Message:
    return Message(
        role="tool",
        content=result.content,
        tool_call_id=result.call.id,
        name=result.call.name,
    )


def _render_calls(calls: list[ToolCall]) -> str:
    return "\n".join(
        f"{call.name}({call.arguments})" for call in calls
    )
