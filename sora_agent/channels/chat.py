"""A minimal chat handler: keep per-session history, run the model/tool loop."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..agent import MEMORY_POLICY, TOOLS_POLICY
from ..eval import (
    QualityGate,
    TrajectoryMetrics,
    trajectory_from_run,
    trajectory_metrics,
)
from ..providers import Message
from ..runner import RunEvent, RunMetrics, RunResult
from ..runtime import AgentRuntime, RunRequest, wrap_identity
from ..tool_search import FIND_TOOLS_NAME
from ..tools import ToolRegistry
from .approval import ApprovalBroker
from .sessions import SessionStore
from .types import (
    STREAM_DELTA_KIND,
    STREAM_END_KIND,
    InboundMessage,
    OutboundMessage,
    TurnReply,
)

log = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = (
    "You are a careful assistant talking with a user over a chat channel. "
    "Answer directly and concisely. Use the tools available to you when they "
    "help, and never claim an action succeeded unless a tool confirmed it."
)

_STREAM_INTERVAL = 0.25
_STREAM_MIN_CHARS = 16


@dataclass
class ChatState:
    """The little bit of run state the memory hooks read."""

    session_id: str
    state: str = ""


class CancelToken:
    """Cooperative stop flag shared with the runner and providers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class _StreamWriter:
    """Coalesce provider chunks into cumulative snapshots for the channel.

    One live message is kept per run step, so progress text from a tool-calling
    turn stays separate from the final answer.
    """

    def __init__(
        self,
        inbound: InboundMessage,
        emit: Callable[[OutboundMessage], None],
        limit: int,
    ) -> None:
        self.inbound = inbound
        self.emit = emit
        self.limit = limit
        self.run_id = uuid.uuid4().hex[:8]
        self.buffers: dict[int, str] = {}
        self.emitted: dict[int, str] = {}
        self.last_emit: dict[int, float] = {}
        self.step = -1

    def delta(self, chunk: str, step: int) -> None:
        if not chunk:
            return
        if step != self.step:
            self._flush(self.step, force=True)
            self._end(self.step)
            self.step = step
        self.buffers[step] = self.buffers.get(step, "") + chunk
        self._flush(step)

    def finish(self, run: RunResult) -> bool:
        """Flush every step and report whether the final answer is on screen."""
        for step in sorted(self.buffers):
            self._flush(step, force=True)
            self._end(step)
        final = run.final_answer.strip()
        if not final or not run.steps:
            return False
        if any(len(text) > self.limit for text in self.buffers.values()):
            # The live message was never rendered in full; let the manager
            # deliver the answer with normal chunking.
            return False
        last = run.steps[-1].index
        return self.buffers.get(last, "").strip() == final

    def _flush(self, step: int, force: bool = False) -> None:
        if step < 0:
            return
        text = self.buffers.get(step, "")
        if not text or text == self.emitted.get(step):
            return
        if len(text) > self.limit:
            return
        now = time.monotonic()
        previous = self.emitted.get(step, "")
        if not force:
            elapsed = now - self.last_emit.get(step, 0.0)
            if elapsed < _STREAM_INTERVAL and len(text) - len(previous) < (
                _STREAM_MIN_CHARS
            ):
                return
        self.emitted[step] = text
        self.last_emit[step] = now
        self.emit(self._message(text, STREAM_DELTA_KIND, step))

    def _end(self, step: int) -> None:
        if step < 0 or step not in self.emitted:
            return
        self.emit(self._message("", STREAM_END_KIND, step))

    def _message(self, text: str, kind: str, step: int) -> OutboundMessage:
        return OutboundMessage(
            channel=self.inbound.channel,
            chat_id=self.inbound.chat_id,
            content=text,
            metadata={
                "kind": kind,
                "stream_id": f"{self.inbound.session_key}:{self.run_id}:{step}",
            },
        )


class ChatHandler:
    """Turn inbound chat messages into replies.

    The gateway runs this from a worker thread, but calls are serialized by a
    lock so one session's history cannot interleave with another's.
    """

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry | None = None,
        *,
        memory: Any = None,
        identity: Any = None,
        approvals: ApprovalBroker | None = None,
        sessions: SessionStore | None = None,
        quality: QualityGate | None = None,
        emitter: Callable[[OutboundMessage], None] | None = None,
        max_steps: int = 8,
        max_turns: int = 40,
        max_tool_result_chars: int = 4000,
        tool_timeout: float = 30.0,
        provider_retries: int = 2,
        on_event: Callable[[RunEvent], None] | None = None,
        max_stream_chars: int = 3500,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()
        if self.registry.has(FIND_TOOLS_NAME):
            raise ValueError(
                f"the tool name {FIND_TOOLS_NAME!r} is reserved for lazy "
                "tool discovery"
            )
        self.memory = memory
        self.identity = None if identity is False else identity
        self.approvals = approvals
        self.sessions = sessions
        self.quality = quality or QualityGate()
        self.emitter = emitter
        self.max_steps = max(1, int(max_steps))
        self.max_turns = max(0, int(max_turns))
        self.max_tool_result_chars = max(0, int(max_tool_result_chars))
        self.tool_timeout = max(0.1, float(tool_timeout))
        self.provider_retries = max(0, int(provider_retries))
        self.on_event = on_event
        self.max_stream_chars = max(1, int(max_stream_chars))
        self.system_prompt = system_prompt
        if self.identity is not None:
            self._attach_identity(self.identity, memory)
            self.provider = wrap_identity(self.identity, provider)
        self.memory_tool: Any = None
        if memory is not None:
            try:
                self.memory_tool = memory.tool()
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory tool registration failed: %s", exc)
        self._history: dict[str, list[Message]] = {}
        self._active: dict[str, CancelToken] = {}
        self._last_metrics: dict[str, RunMetrics] = {}
        self._last_stop: dict[str, str] = {}
        self._last_report: dict[str, Any] = {}
        self._last_efficiency: dict[str, TrajectoryMetrics] = {}
        self._lock = threading.Lock()

    def handle(self, inbound: InboundMessage) -> str | TurnReply:
        """Run one turn and return the text to send back."""
        key = inbound.session_key
        token = CancelToken()
        with self._lock:
            history = self._load_history(key)
            self._active[key] = token

        approval = (
            self.approvals.gate(inbound) if self.approvals is not None else None
        )
        writer = (
            _StreamWriter(inbound, self.emitter, self.max_stream_chars)
            if self.emitter is not None
            else None
        )

        def on_delta(chunk: str, step: int) -> None:
            if writer is not None and not token.cancelled:
                writer.delta(chunk, step)

        messages = [
            Message("system", self._prompt()),
            *history,
            Message("user", inbound.content),
        ]
        state = ChatState(session_id=key)
        self._notify("before_run", inbound.content, state)
        try:
            run = AgentRuntime.run(
                RunRequest(
                    provider=self.provider,
                    registry=self.registry,
                    messages=messages,
                    memory_tool=self.memory_tool,
                    approval=approval,
                    max_steps=self.max_steps,
                    max_result_chars=self.max_tool_result_chars,
                    tool_timeout=self.tool_timeout,
                    provider_retries=self.provider_retries,
                    context=f"channel={inbound.channel}",
                    on_delta=on_delta if writer is not None else None,
                    should_stop=lambda: token.cancelled,
                    on_event=self.on_event,
                )
            )
        finally:
            with self._lock:
                if self._active.get(key) is token:
                    del self._active[key]
        self._notify("after_run", run, state)

        reply = _reply_text(run)
        trajectory = trajectory_from_run(
            inbound.content, run, final_answer=reply
        )
        report = self.quality.evaluate(trajectory)
        efficiency = trajectory_metrics(trajectory)
        with self._lock:
            self._last_metrics[key] = run.metrics
            self._last_stop[key] = run.stop_reason
            self._last_report[key] = report
            self._last_efficiency[key] = efficiency

        delivered = writer.finish(run) if writer is not None else False
        stored = list(history)
        stored.append(Message("user", inbound.content))
        stored.append(Message("assistant", reply))
        limit = self.max_turns * 2
        if limit and len(stored) > limit:
            del stored[: len(stored) - limit]
        with self._lock:
            self._store_history(key, stored)
        if writer is None:
            return reply
        return TurnReply(reply, delivered=delivered, report=report)

    def cancel(self, session_key: str) -> bool:
        """Ask a running turn to stop.  Returns whether one was running."""
        with self._lock:
            token = self._active.get(session_key)
        if token is None:
            return False
        token.cancel()
        if self.approvals is not None:
            # A turn blocked on a human must wake now, not at the timeout.
            self.approvals.cancel(session_key)
        return True

    def cancel_all(self) -> int:
        """Cancel every running turn, for a clean gateway shutdown."""
        with self._lock:
            tokens = list(self._active.values())
        for token in tokens:
            token.cancel()
        if self.approvals is not None:
            self.approvals.cancel_all()
        return len(tokens)

    def handle_command(self, inbound: InboundMessage) -> str | None:
        """Answer a control command, or return ``None`` to let the agent run."""
        text = inbound.content.strip()
        if not text.startswith("/"):
            return None
        command = text.split()[0].lower().split("@", 1)[0]
        if command == "/cancel":
            if self.cancel(inbound.session_key):
                return "Cancelled. I stopped that turn."
            return "Nothing is running right now."
        if command == "/reset":
            self.reset(inbound.session_key)
            return "Conversation reset."
        if command == "/status":
            return self._status(inbound.session_key)
        if command == "/eval":
            return self.eval_report(inbound.session_key)
        return None

    def last_report(self, session_key: str):
        """Return the latest evaluation report, if any."""
        with self._lock:
            return self._last_report.get(session_key)

    def eval_report(self, session_key: str) -> str:
        report = self.last_report(session_key)
        if report is None:
            return "No evaluation for this session yet."
        return str(report)

    def _status(self, session_key: str) -> str:
        with self._lock:
            metrics = self._last_metrics.get(session_key)
            stop = self._last_stop.get(session_key, "")
            report = self._last_report.get(session_key)
            efficiency = self._last_efficiency.get(session_key)
        if metrics is None:
            return "No run in this session yet."
        tokens = (
            f"{metrics.total_tokens} tokens"
            if metrics.total_tokens
            else "token usage unavailable"
        )
        extras: list[str] = []
        if efficiency is not None:
            extras.append(
                f"ASR {efficiency.successful_tool_calls}/"
                f"{efficiency.attempted_tool_calls}"
            )
            if efficiency.redundant_call_rate:
                extras.append(
                    f"redundant {efficiency.redundant_call_rate:.0%}"
                )
        if report is not None:
            if report.error:
                extras.append("eval error")
            else:
                verdict = (
                    "skipped"
                    if report.skipped
                    else ("pass" if report.passed else "fail")
                )
                extras.append(f"eval {verdict} {report.score:.2f}")
        suffix = ("; " + ", ".join(extras)) if extras else ""
        return (
            f"Last run: {stop or 'unknown'}, {metrics.steps} steps, "
            f"{metrics.tool_calls} tool calls "
            f"({metrics.tool_errors} errors), "
            f"{metrics.duration:.1f}s, {tokens}{suffix}"
        )

    def reset(self, session_key: str) -> None:
        """Forget one conversation, on disk as well as in memory."""
        with self._lock:
            self._history.pop(session_key, None)
            if self.sessions is not None:
                self.sessions.clear(session_key)

    def _load_history(self, key: str) -> list[Message]:
        if self.sessions is not None:
            return self.sessions.load(key)
        return list(self._history.get(key, ()))

    def _store_history(self, key: str, messages: list[Message]) -> None:
        if self.sessions is not None:
            self.sessions.save(key, messages)
        else:
            self._history[key] = messages

    def _prompt(self) -> str:
        prompt = self.system_prompt + TOOLS_POLICY
        if self.memory_tool is not None:
            prompt += MEMORY_POLICY
        return prompt

    @staticmethod
    def _attach_identity(identity: Any, memory: Any) -> None:
        setter = getattr(memory, "set_identity", None)
        if not callable(setter):
            return
        try:
            setter(identity)
        except Exception as exc:  # noqa: BLE001 - identity is optional
            log.warning("memory identity attachment failed: %s", exc)

    def _notify(self, name: str, *args: Any) -> None:
        if self.memory is None:
            return
        callback = getattr(self.memory, name, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:  # noqa: BLE001 - memory must fail open
            log.warning("memory %s failed: %s", name, exc)


def _reply_text(run: RunResult) -> str:
    text = run.final_answer.strip()
    if text:
        return text
    if run.stop_reason == "approval_rejected":
        return "I didn't run that: it needs human approval, and none was given."
    if run.stop_reason == "cancelled":
        return "Cancelled."
    for step in reversed(run.steps):
        candidate = step.turn.text.strip()
        if candidate:
            return candidate
    if run.error:
        return f"I couldn't finish that: {run.error}"
    return "I couldn't produce a reply for that message."
