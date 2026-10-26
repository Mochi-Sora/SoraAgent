"""The agent loop: plan -> grade -> (approve) -> execute -> observe.

Kept intentionally small. The interesting parts for production are the **quality gate**
(does this pass our bar?) and the **approval gate** (should a human sign off?) — both
inspectable and swappable.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .approval import ApprovalGate
from .eval import QualityGate
from .providers import Message, ModelProvider
from .tools import ToolRegistry

SYSTEM_PROMPT = (
    "You are a careful enterprise-workflow agent. You may call tools to complete a task, and "
    "you MUST NOT perform a consequential action without approval.\n"
    "When you want to call a tool, output exactly one line like:\n"
    'ACTION: tool_name(arg="value")\n'
    "Then briefly explain in plain language."
)

_ACTION_RE = re.compile(r"ACTION:\s*([a-zA-Z_]+)\s*\((.*)\)", re.DOTALL)


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


class Agent:
    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
        approval: ApprovalGate | None = None,
        quality: QualityGate | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.approval = approval or ApprovalGate()
        self.quality = quality or QualityGate()

    def run(self, task: str, state: Any = None) -> AgentResult:
        # 1) plan
        messages = [
            Message("system", SYSTEM_PROMPT),
            Message("user", task),
        ]
        proposal = self.provider.complete(messages, self.registry.all())
        result = AgentResult(task=task, proposal=proposal)

        # 2) quality gate
        result.report = self.quality.grade(proposal, f"state={getattr(state, 'state', '')}")
        if not result.report.passed:
            self.approval.log("quality-gate", "blocked", result.report.__str__())
            return result

        # 3) resolve tool call
        tool_name, kwargs = self._parse_action(proposal)
        if not tool_name:
            self.approval.log("no-action", "blocked", "no parseable ACTION line")
            return result
        result.action = tool_name
        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            result.tool_result = f"unknown tool: {tool_name}"
            return result

        # 4) approval gate for consequential actions
        if tool.needs_approval:
            detail = f"{tool_name}({kwargs})"
            result.approved = self.approval.request(tool_name, detail)
            if not result.approved:
                result.tool_result = "rejected by human approval gate"
                return result

        # 5) execute
        try:
            result.tool_result = str(tool.call(**kwargs))
        except TypeError as exc:
            result.tool_result = f"tool call failed: {exc}"

        # 6) advance state machine if provided
        if state is not None:
            target = self._state_target(tool_name, result.proposal)
            if target:
                try:
                    state.transition(target, reason=tool_name)
                except Exception as exc:  # noqa: BLE001 - surface any transition error
                    result.tool_result += f" | state: {exc}"
            result.state = getattr(state, "state", "")

        result.audit = self.approval.audit
        self.approval.log(tool_name, "executed" if result.approved or not tool.needs_approval
                          else "rejected", result.tool_result)
        return result

    @staticmethod
    def _parse_action(proposal: str) -> tuple[str, dict[str, Any]]:
        m = _ACTION_RE.search(proposal)
        if not m:
            return "", {}
        name = m.group(1)
        raw = m.group(2)
        return name, Agent._parse_kwargs(raw)

    @staticmethod
    def _parse_kwargs(raw: str) -> dict[str, Any]:
        """Parse `k="v", n=3` into a dict (quote-aware top-level comma split)."""
        kwargs: dict[str, Any] = {}
        if not raw.strip():
            return kwargs
        for part in Agent._split_top(raw, ","):
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            val = val.strip()
            try:
                kwargs[key.strip()] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                kwargs[key.strip()] = val.strip("\"'")
        return kwargs

    @staticmethod
    def _split_top(s: str, sep: str) -> list[str]:
        out: list[str] = []
        buf: list[str] = []
        depth = 0
        quote = ""
        for ch in s:
            if ch == quote:
                quote = ""
                buf.append(ch)
                continue
            if not quote:
                if ch == sep and depth == 0:
                    out.append("".join(buf))
                    buf = []
                    continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch in "\"'":
                    quote = ch
            buf.append(ch)
        if buf:
            out.append("".join(buf))
        return out

    @staticmethod
    def _state_target(tool_name: str, proposal: str) -> Any:
        from .state_machine import State

        mapping = {
            "expedite_order": State.EXPEDITED,
            "cancel_order": State.CANCELLED,
            "defer_order": State.DEFERRED,
            "reroute_order": State.REROUTED,
        }
        return mapping.get(tool_name)
