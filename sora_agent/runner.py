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

#: Corrections sent when the model skips the required first tool.  A third
#: failure hard-stops the run instead of letting it continue without the tool.
_MAX_FIRST_TOOL_CORRECTIONS = 2


@dataclass
class Step:
    """One assistant turn and the tool results it produced."""

    index: int
    turn: ModelTurn
    results: list[ToolResult] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


@dataclass
class RunSpec:
    messages: list[Message]
    tools: list[Any]
    max_steps: int = 10
    context: str = ""
    # Called once per action-bearing turn, before any tool runs. Returns
    # (passed, report). A failed guard stops the run with quality_blocked.
    guard: Callable[[str, str], tuple[bool, Any]] | None = None
    # Recompute the visible tool list for every model step (lazy discovery).
    tool_provider: Callable[[], list[Any]] | None = None
    # When set, the model's first action must be this tool.  Text reasoning is
    # allowed, but a direct answer or another tool call is rejected and retried.
    required_first_tool: str | None = None


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
        required = str(spec.required_first_tool or "").strip()
        required_pending = bool(required)
        corrections = 0

        for index in range(max(0, int(spec.max_steps))):
            try:
                tools = (
                    spec.tool_provider()
                    if spec.tool_provider is not None
                    else spec.tools
                )
                turn = provider_turn(self.provider, messages, tools)
            except Exception as exc:  # noqa: BLE001 - surface provider failures
                stop_reason = "provider_error"
                error = f"{type(exc).__name__}: {exc}"
                break

            visible = [getattr(tool, "name", "") for tool in tools]
            calls = list(turn.tool_calls)
            recall_turn = False

            if required_pending:
                if not calls or calls[0].name != required:
                    messages.append(
                        Message(
                            role="assistant",
                            content=turn.text,
                            tool_calls=calls,
                        )
                    )
                    results: list[ToolResult] = []
                    if calls:
                        results = _skipped_results(
                            calls,
                            f"Skipped: your first action must be "
                            f"`{required}`; nothing was executed.",
                        )
                        for result in results:
                            messages.append(_tool_message(result))
                    corrections += 1
                    steps.append(
                        Step(
                            index=index,
                            turn=turn,
                            results=results,
                            tools=visible,
                        )
                    )
                    if corrections > _MAX_FIRST_TOOL_CORRECTIONS:
                        stop_reason = f"{required}_required"
                        break
                    messages.append(
                        Message("user", _first_tool_correction(required))
                    )
                    continue
                required_pending = False
                recall_turn = True

            if not calls:
                messages.append(Message("assistant", turn.text))
                steps.append(Step(index=index, turn=turn, tools=visible))
                final_answer = turn.text
                stop_reason = "completed"
                break

            messages.append(
                Message(
                    role="assistant",
                    content=turn.text,
                    tool_calls=calls,
                )
            )

            # Recall is a read-only prelude; quality policy applies to the
            # consequential actions that follow.
            if spec.guard is not None and not recall_turn:
                proposal = "\n".join(
                    part
                    for part in (
                        turn.text.strip(),
                        _render_calls(calls),
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
                    results = _blocked_results(calls, guard_report)
                    for result in results:
                        messages.append(_tool_message(result))
                    steps.append(
                        Step(
                            index=index,
                            turn=turn,
                            results=results,
                            tools=visible,
                        )
                    )
                    stop_reason = "quality_blocked"
                    break

            execute_calls = calls[:1] if recall_turn else calls
            results = []
            stopped = ""
            for call in execute_calls:
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
                for call in calls[len(results):]:
                    messages.append(
                        Message(
                            role="tool",
                            content=f"Skipped: run stopped ({stopped}).",
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
            elif recall_turn and len(calls) > 1:
                # The first action completed; the rest of that turn is skipped
                # so the model can see the required result before acting.
                for result in _skipped_results(
                    calls[1:],
                    f"Skipped: `{required}` must complete before other tools.",
                ):
                    results.append(result)
                    messages.append(_tool_message(result))

            steps.append(
                Step(
                    index=index,
                    turn=turn,
                    results=results,
                    tools=visible,
                )
            )
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


def _skipped_results(
    calls: list[ToolCall], content: str
) -> list[ToolResult]:
    return [
        ToolResult(call=call, content=content, is_error=True, skipped=True)
        for call in calls
    ]


def _first_tool_correction(required: str) -> str:
    return (
        "Rejected: your first action must be "
        f"`{required}`. Do not answer, and do not call another tool first. "
        "Think if you need to, then call it."
    )


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
