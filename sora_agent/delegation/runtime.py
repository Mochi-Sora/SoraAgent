"""Runtime glue for finding and invoking specialists."""

from __future__ import annotations

from typing import Protocol

from .model import DelegationAgent
from .router import rank
from .store import DelegationStore


class SpecialistRunner(Protocol):
    def __call__(
        self,
        agent: DelegationAgent,
        goal: str,
        context: str = "",
        *,
        approval: object = None,
    ) -> str: ...


class UnavailableSpecialistRunner:
    def __call__(
        self, agent: DelegationAgent, goal: str, context: str = ""
    ) -> str:
        return (
            "error: specialists are not wired on this surface; no specialist "
            "runner was supplied"
        )


class DelegationRuntime:
    def __init__(
        self,
        store: DelegationStore | None = None,
        runner: SpecialistRunner | None = None,
    ) -> None:
        self.store = store or DelegationStore()
        self.runner: SpecialistRunner = runner or UnavailableSpecialistRunner()

    def set_runner(self, runner: SpecialistRunner | None) -> None:
        self.runner = runner or UnavailableSpecialistRunner()

    def agents(self, *, include_paused: bool = False) -> list[DelegationAgent]:
        return self.store.list(include_paused=include_paused)

    def find(
        self, need: str, limit: int = 3, *, include_paused: bool = False
    ) -> list[tuple[DelegationAgent, float]]:
        return rank(self.agents(include_paused=include_paused), need, limit=limit)

    def run(
        self, agent_id: str, goal: str, context: str = ""
    ) -> str:
        agent_id = str(agent_id or "").strip()
        goal = str(goal or "").strip()
        if not agent_id:
            return "error: provide a specialist id"
        if not goal:
            return "error: provide a goal for the specialist"
        agent = self.store.get(agent_id)
        if agent is None:
            available = ", ".join(item.id for item in self.agents()) or "none"
            return f"error: unknown specialist {agent_id!r} (available: {available})"
        if agent.status != "active":
            return f"error: specialist {agent_id!r} is {agent.status}"
        try:
            result = self.runner(agent, goal, context)
        except Exception as exc:  # noqa: BLE001 - a child failure is data
            return f"error: specialist {agent_id!r} failed: {exc}"
        text = str(result or "").strip()
        if not text:
            return f"error: specialist {agent_id!r} returned no result"
        return f"Specialist `{agent.id}` ({agent.name}) result:\n\n{text}"


def format_find_results(
    matches: list[tuple[DelegationAgent, float]], need: str
) -> str:
    if not matches:
        return (
            f"No specialist matched {need!r}. The user can create one with "
            "`sora delegate create`; otherwise use an ephemeral sub-agent."
        )
    lines = [
        (
            f"Specialists matching {need!r} "
            "(call `delegate_to_agent` with the id):"
        ),
        "",
    ]
    for agent, score in matches:
        grants = ", ".join(agent.resolved_tools())
        model = f"{agent.provider}/{agent.model}".strip("/") or "main model"
        lines.append(
            f"- `{agent.id}` — **{agent.name}**: "
            f"{agent.description or agent.when_to_use}"
        )
        if agent.when_to_use:
            lines.append(f"  Use when: {agent.when_to_use}")
        lines.append(f"  Tools: {grants}; model: {model}; score: {score:.2f}")
    return "\n".join(lines)
