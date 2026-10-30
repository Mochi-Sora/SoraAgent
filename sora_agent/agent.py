"""The agent facade: assemble the turn, run the loop, map the result.

The interesting production concerns live in the gates: the **quality gate** (does
this pass our bar?) and the **approval gate** (should a human sign off?).  The
model/tool loop itself is in ``runner.py`` and stays deliberately small.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .approval import ApprovalGate
from .eval import QualityGate
from .providers import Message, ModelProvider, ToolCall
from .runner import RunResult, Runner, RunSpec, Step
from .tools import ToolExecutor, ToolRegistry, ToolResult

SYSTEM_PROMPT = (
    "You are a careful enterprise-workflow agent. Use the tools available to complete "
    "the task, and you MUST NOT perform a consequential action without approval.\n"
    "If native tool calls are unavailable, call one tool by outputting exactly one line "
    "like:\n"
    'ACTION: tool_name(arg="value")\n'
    "Then briefly explain in plain language."
)

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    task: str
    proposal: str
    report: Any = None          # EvalReport
    action: str = ""
    tool_result: str = ""
    state: str = ""
    approved: bool = False
    audit: list[Any] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = ""
    steps: list[Step] = field(default_factory=list)
    error: str = ""


class Agent:
    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
        approval: ApprovalGate | None = None,
        quality: QualityGate | None = None,
        memory: Any = None,
        identity: Any = None,
        max_steps: int = 10,
        max_tool_result_chars: int = 4000,
    ) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.approval = approval or ApprovalGate()
        self.quality = quality or QualityGate()
        self.max_steps = max(0, int(max_steps))
        self.max_tool_result_chars = max(0, int(max_tool_result_chars))
        self.identity = self._resolve_identity(identity)
        self.memory = self._resolve_memory(memory, self.identity)
        if self.memory is not None:
            try:
                self.provider = self.memory.wrap_provider(self.provider)
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory provider wrapper failed: %s", exc)
            try:
                self.registry.add(self.memory.tool())
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory tool registration failed: %s", exc)
        if self.identity is not None and not getattr(
            self.provider, "__sora_identity__", False
        ):
            try:
                self.provider = self.identity.wrap_provider(self.provider)
            except Exception as exc:  # noqa: BLE001 - identity is optional
                log.warning("identity provider wrapper failed: %s", exc)

    def run(self, task: str, state: Any = None) -> AgentResult:
        self._notify("before_run", task, state)
        result: AgentResult | None = None
        try:
            result = self._run(task, state)
            return result
        finally:
            if result is not None:
                self._notify("after_run", result, state)

    def _run(self, task: str, state: Any = None) -> AgentResult:
        messages = [
            Message("system", SYSTEM_PROMPT),
            Message("user", task),
        ]
        executor = ToolExecutor(
            registry=self.registry,
            approval=self.approval,
            max_result_chars=self.max_tool_result_chars,
            after_execute=lambda call, result: self._after_tool(
                state, call, result
            ),
        )

        def guard(proposal: str, _context: str) -> tuple[bool, Any]:
            return self._quality_guard(
                proposal,
                f"state={getattr(state, 'state', '')}",
            )

        run = Runner(self.provider, executor).run(
            RunSpec(
                messages=messages,
                tools=self.registry.all(),
                max_steps=self.max_steps,
                context=f"state={getattr(state, 'state', '')}",
                guard=guard,
            )
        )
        return self._to_result(task, state, run)

    def _to_result(
        self, task: str, state: Any, run: RunResult
    ) -> AgentResult:
        proposal = run.final_answer or self._last_text(run)
        result = AgentResult(
            task=task,
            proposal=proposal,
            final_answer=run.final_answer,
            stop_reason=run.stop_reason,
            steps=run.steps,
            error=run.error,
        )

        results = run.tool_results
        if results:
            last = results[-1]
            result.action = last.call.name
            result.tool_result = last.content
            result.approved = any(
                item.approved
                for item in results
                if item.approval_required
            )
        if run.report is not None:
            result.report = run.report
        else:
            result.report = next(
                (
                    item.report
                    for item in reversed(results)
                    if item.report is not None
                ),
                None,
            )
        if state is not None:
            result.state = getattr(state, "state", "")
        result.audit = self.approval.audit
        return result

    def _quality_guard(
        self, proposal: str, context: str
    ) -> tuple[bool, Any]:
        report = self.quality.grade(proposal, context)
        return bool(getattr(report, "passed", False)), report

    def _after_tool(
        self, state: Any, call: ToolCall, result: ToolResult
    ) -> None:
        if state is None or not result.executed:
            return
        target = self._state_target(call.name)
        if target is None:
            return
        try:
            state.transition(target, reason=call.name)
        except Exception as exc:  # noqa: BLE001 - surface any transition error
            result.content += f" | state: {exc}"

    @staticmethod
    def _last_text(run: RunResult) -> str:
        for step in reversed(run.steps):
            if step.turn.text:
                return step.turn.text
        return ""

    @staticmethod
    def _resolve_identity(identity: Any) -> Any:
        if identity is False:
            return None
        if identity is None or identity is True or identity == "default":
            try:
                from .identity import default_identity

                return default_identity()
            except Exception as exc:  # noqa: BLE001 - identity is optional
                log.warning("identity initialization failed: %s", exc)
                return None
        return identity

    @staticmethod
    def _resolve_memory(memory: Any, identity: Any = None) -> Any:
        if memory is False:
            return None
        if memory is None or memory is True or memory == "default":
            try:
                from .memory import default_memory

                return default_memory(identity=identity)
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory initialization failed: %s", exc)
                return None
        if identity is not None:
            setter = getattr(memory, "set_identity", None)
            if callable(setter):
                try:
                    setter(identity)
                except Exception as exc:  # noqa: BLE001 - memory is optional
                    log.warning("memory identity attachment failed: %s", exc)
        return memory

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

    @staticmethod
    def _state_target(tool_name: str) -> Any:
        from .state_machine import State

        mapping = {
            "expedite_order": State.EXPEDITED,
            "cancel_order": State.CANCELLED,
            "defer_order": State.DEFERRED,
            "reroute_order": State.REROUTED,
        }
        return mapping.get(tool_name)
