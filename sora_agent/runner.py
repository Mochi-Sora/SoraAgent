"""The pure model/tool loop.

``Runner`` knows nothing about sessions, files, identity, state, or the CLI.  It
receives messages and tools, asks the provider for a turn, hands tool calls to an
executor, appends the observations, and repeats until the model answers, a policy
stops the run, or the step budget is exhausted.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .providers import Message, ModelTurn, ToolCall, provider_turn
from .tools import ToolExecutor, ToolResult

log = logging.getLogger(__name__)

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
class RunEvent:
    """One observable moment in a run."""

    kind: str
    step: int = -1
    name: str = ""
    duration: float = 0.0
    ok: bool = True
    detail: str = ""


@dataclass
class RunMetrics:
    """Deterministic accounting for one run."""

    duration: float = 0.0
    steps: int = 0
    provider_calls: int = 0
    provider_retries: int = 0
    provider_duration: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    tool_duration: float = 0.0
    streamed_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
    # Called with (chunk, step_index) as the model produces text.
    on_delta: Callable[[str, int], None] | None = None
    # Cooperative cancellation, checked between steps and between tool calls.
    should_stop: Callable[[], bool] | None = None
    # Transient provider failures are retried with exponential backoff and
    # full jitter.  A retry never repeats a turn that already streamed text.
    provider_retries: int = 2
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    # When set, the model's first action must be this tool.  Text reasoning is
    # allowed, but a direct answer or another tool call is rejected and retried.
    required_first_tool: str | None = None
    # Called for each observable moment.  An observer must never affect a run.
    on_event: Callable[[RunEvent], None] | None = None


@dataclass
class RunResult:
    messages: list[Message]
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = "max_steps"
    report: Any = None
    error: str = ""
    metrics: RunMetrics = field(default_factory=RunMetrics)

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
        started = time.monotonic()
        metrics = RunMetrics()

        def cancelled() -> bool:
            return spec.should_stop is not None and spec.should_stop()

        def notify(event: RunEvent) -> None:
            if spec.on_event is None:
                return
            try:
                spec.on_event(event)
            except Exception:
                log.debug("run event handler failed", exc_info=True)

        notify(RunEvent("run_started"))

        for index in range(max(0, int(spec.max_steps))):
            if cancelled():
                stop_reason = "cancelled"
                break
            try:
                tools = (
                    spec.tool_provider()
                    if spec.tool_provider is not None
                    else spec.tools
                )
            except Exception as exc:  # noqa: BLE001 - surface provider failures
                stop_reason = "provider_error"
                error = f"{type(exc).__name__}: {exc}"
                break

            provider_started = time.monotonic()
            turn, provider_error, was_cancelled = _turn_with_retries(
                self.provider, spec, messages, tools, index, metrics, notify
            )
            provider_duration = time.monotonic() - provider_started
            metrics.provider_duration += provider_duration
            if turn is None:
                stop_reason = "cancelled" if was_cancelled else "provider_error"
                error = provider_error
                notify(
                    RunEvent(
                        "provider_finished",
                        step=index,
                        duration=provider_duration,
                        ok=False,
                        detail=provider_error,
                    )
                )
                break
            notify(
                RunEvent(
                    "provider_finished",
                    step=index,
                    duration=provider_duration,
                )
            )
            if turn.usage is not None:
                metrics.prompt_tokens += turn.usage.prompt_tokens
                metrics.completion_tokens += turn.usage.completion_tokens
                metrics.total_tokens += turn.usage.total_tokens

            if cancelled():
                stop_reason = "cancelled"
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
                if cancelled():
                    stopped = "cancelled"
                    break
                tool_started = time.monotonic()
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
                tool_duration = time.monotonic() - tool_started
                metrics.tool_calls += 1
                metrics.tool_duration += tool_duration
                if result.is_error:
                    metrics.tool_errors += 1
                notify(
                    RunEvent(
                        "tool_finished",
                        step=index,
                        name=call.name,
                        duration=tool_duration,
                        ok=not result.is_error,
                        detail=result.stop_reason,
                    )
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

        metrics.duration = time.monotonic() - started
        metrics.steps = len(steps)
        notify(
            RunEvent(
                "run_finished",
                duration=metrics.duration,
                ok=stop_reason == "completed",
                detail=stop_reason,
            )
        )
        return RunResult(
            messages=messages,
            steps=steps,
            final_answer=final_answer,
            stop_reason=stop_reason,
            report=report,
            error=error,
            metrics=metrics,
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


class _DeltaCounter:
    """Count streamed chunks so a retry never repeats visible text."""

    def __init__(self) -> None:
        self.count = 0
        self.chars = 0

    def wrap(self, sink: Callable[[str], None]) -> Callable[[str], None]:
        def emit(chunk: str) -> None:
            self.count += 1
            self.chars += len(chunk)
            sink(chunk)

        return emit


def _turn_with_retries(
    provider: Any,
    spec: RunSpec,
    messages: list[Message],
    tools: list[Any],
    step: int,
    metrics: RunMetrics,
    notify: Callable[[RunEvent], None],
) -> tuple[ModelTurn | None, str, bool]:
    """Call the provider, retrying only failures that streamed nothing."""
    retries = max(0, int(spec.provider_retries))
    attempt = 0
    while True:
        metrics.provider_calls += 1
        counter = _DeltaCounter()
        on_delta: Callable[[str], None] | None = None
        if spec.on_delta is not None:
            on_delta = counter.wrap(_bind_delta(spec.on_delta, step))
        try:
            turn = provider_turn(
                provider,
                messages,
                tools,
                on_delta=on_delta,
                should_stop=spec.should_stop,
            )
            metrics.streamed_chars += counter.chars
            return turn, "", False
        except Exception as exc:  # noqa: BLE001 - retry policy lives here
            cancelled = spec.should_stop is not None and spec.should_stop()
            if attempt >= retries or counter.count > 0 or cancelled:
                return None, f"{type(exc).__name__}: {exc}", cancelled
            metrics.provider_retries += 1
            notify(
                RunEvent(
                    "provider_retry",
                    step=step,
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            delay = _retry_delay(spec, attempt)
            log.warning(
                "provider failed (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1,
                retries + 1,
                exc,
                delay,
            )
            _sleep_unless_cancelled(delay, spec.should_stop)
            if spec.should_stop is not None and spec.should_stop():
                return None, f"{type(exc).__name__}: {exc}", True
            attempt += 1


def _retry_delay(spec: RunSpec, attempt: int) -> float:
    base = max(0.0, float(spec.retry_base_delay))
    cap = max(0.0, float(spec.retry_max_delay))
    window = min(cap, base * (2**attempt)) if base else 0.0
    if window <= 0:
        return 0.0
    # Equal jitter keeps every retry at least half the window, so a retry never
    # collapses to a hot loop.
    return random.uniform(window * 0.5, window)


def _sleep_unless_cancelled(
    delay: float, should_stop: Callable[[], bool] | None
) -> None:
    deadline = time.monotonic() + max(0.0, delay)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if should_stop is not None and should_stop():
            return
        time.sleep(min(0.05, remaining))


def _bind_delta(
    sink: Callable[[str, int], None], step: int
) -> Callable[[str], None]:
    def emit(chunk: str) -> None:
        sink(chunk, step)

    return emit


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
