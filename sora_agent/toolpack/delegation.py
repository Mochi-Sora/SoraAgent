"""Subagent delegation tool.

The tool is deliberately runtime-agnostic.  A caller supplies a runner that
executes one isolated goal and returns a summary; the pack does not create a
provider or share the parent conversation.
"""

from __future__ import annotations

from typing import Annotated, Any, Protocol

from ..tools import Tool, ToolRegistry


class DelegateRunner(Protocol):
    def __call__(self, goal: str) -> str: ...


class UnavailableDelegateRunner:
    def __call__(self, goal: str) -> str:
        return (
            "error: delegation is not configured on this surface; no subagent "
            "runner was supplied"
        )


class DelegationRuntime:
    """Mutable runner slot so an Agent can wire delegation after registry build."""

    def __init__(self, runner: DelegateRunner | None = None) -> None:
        self.runner: DelegateRunner = runner or UnavailableDelegateRunner()

    def run(self, goal: str) -> str:
        return str(self.runner(goal))


def register_delegation_tools(
    registry: ToolRegistry,
    config: Any,
    *,
    delegate_runner: DelegateRunner | None = None,
) -> list[Tool]:
    runtime = DelegationRuntime(delegate_runner)
    added: list[Tool] = []

    def delegate_task(
        goal: Annotated[str, "A self-contained goal for one subagent"] = "",
        tasks: Annotated[
            list[str] | None, "Independent goals to run in parallel (max 4)"
        ] = None,
    ) -> str:
        goals = [str(item).strip() for item in (tasks or []) if str(item).strip()]
        if goal.strip():
            goals.insert(0, goal.strip())
        if not goals:
            return "error: provide goal or tasks"
        goals = goals[:4]
        lines = []
        for index, item in enumerate(goals, 1):
            try:
                result = runtime.run(item)
            except Exception as exc:  # noqa: BLE001 - a child failure is data
                result = f"error: subagent failed: {exc}"
            lines.append(f"Subagent {index}: {item}\n{result}")
        return "\n\n".join(lines)

    definition = Tool(
        delegate_task,
        "delegate_task",
        "Spawn isolated subagents for independent goals and return their summaries.",
        aliases=("delegate", "subagent", "parallel task", "spawn agent"),
    )
    definition.delegation_runtime = runtime  # type: ignore[attr-defined]
    if not registry.has(definition.name):
        registry.add(definition)
        added.append(definition)
    return added
