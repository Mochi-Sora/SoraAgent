"""Tools that expose the specialist registry to the orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from ..tools import Tool, ToolRegistry
from .model import DelegationError
from .runtime import DelegationRuntime, format_find_results
from .store import DelegationStore

SAFE_CREATE_PACKS = frozenset({"research", "files", "web", "planning"})


def _safe_creation_tools(tools: str) -> tuple[tuple[str, ...], list[str]]:
    requested = tuple(item.strip() for item in str(tools or "").split(",") if item.strip())
    safe: list[str] = []
    rejected: list[str] = []
    allowed_names = {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "todo_list",
        "clarify",
        "memory_recall",
    }
    for item in requested:
        if item in SAFE_CREATE_PACKS or item in allowed_names:
            safe.append(item)
        else:
            rejected.append(item)
    return tuple(safe or ("research",)), rejected


def register_specialist_tools(
    registry: ToolRegistry,
    store: DelegationStore | None = None,
    *,
    runner: Any = None,
    allow_create: bool = False,
    max_agents: int = 20,
    max_per_day: int = 3,
) -> list[Tool]:
    runtime = DelegationRuntime(store, runner)
    added: list[Tool] = []

    def find_delegates(
        need: Annotated[str, "Plain-language description of the work"],
        limit: Annotated[int, "Maximum matches (1-10)"] = 3,
    ) -> str:
        matches = runtime.find(need, limit=limit)
        return format_find_results(matches, need)

    def delegate_to_agent(
        agent: Annotated[str, "Specialist id from find_delegates"],
        goal: Annotated[str, "A self-contained goal for the specialist"],
        context: Annotated[
            str, "Relevant context, constraints, or artifact paths"
        ] = "",
    ) -> str:
        return runtime.run(agent, goal, context)

    def create_delegation_agent(
        name: Annotated[str, "Short specialist name"],
        description: Annotated[str, "One-line purpose"],
        when: Annotated[str, "When the orchestrator should use it"],
        prompt: Annotated[str, "The specialist system prompt"],
        tools: Annotated[
            str, "Comma-separated safe packs: research, files, web, planning"
        ] = "research",
    ) -> str:
        if not allow_create:
            return (
                "error: on-demand specialist creation is disabled; the user can "
                "create one with `sora delegate create`"
            )
        if not str(prompt or "").strip():
            return "error: a specialist prompt is required"
        selected, rejected = _safe_creation_tools(tools)
        agents = runtime.agents(include_paused=True)
        if len(agents) >= max_agents:
            return f"error: specialist limit reached ({max_agents}); delete one first"
        today = datetime.now(timezone.utc).date().isoformat()
        created_today = sum(
            1
            for agent in agents
            if agent.created_by == "main"
            and str(agent.metadata.get("created_at", "")).startswith(today)
        )
        if created_today >= max_per_day:
            return f"error: daily specialist creation limit reached ({max_per_day})"
        matches = runtime.find(f"{name} {description} {when}", limit=1)
        if matches and matches[0][1] >= 3.0:
            existing = matches[0][0]
            return (
                f"A similar specialist already exists: `{existing.id}` "
                f"({existing.name}). Use `delegate_to_agent` instead."
            )
        try:
            agent = runtime.store.create(
                name=name,
                description=description,
                when_to_use=when,
                prompt=prompt,
                tools=selected,
                created_by="main",
            )
        except DelegationError as exc:
            return f"error: {exc}"
        agent.metadata["provisional"] = True
        agent.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        runtime.store.save(agent)
        note = (
            f"Created provisional specialist `{agent.id}` ({agent.name}) with "
            f"tools: {', '.join(selected)}."
        )
        if rejected:
            note += " Rejected unsafe tools: " + ", ".join(rejected) + "."
        return note

    find_tool = Tool(
        find_delegates,
        "find_delegates",
        "Find user-created specialists whose skills match a task.",
        aliases=("specialist", "find specialist", "delegation agent", "expert"),
    )
    delegate_tool = Tool(
        delegate_to_agent,
        "delegate_to_agent",
        "Hand a self-contained task to a user-created specialist.",
        aliases=("delegate to specialist", "ask specialist", "hand off"),
    )
    find_tool.delegation_runtime = runtime  # type: ignore[attr-defined]
    delegate_tool.delegation_runtime = runtime  # type: ignore[attr-defined]
    tools_to_add = [find_tool, delegate_tool]
    if allow_create:
        create_tool = Tool(
            create_delegation_agent,
            "create_delegation_agent",
            "Create a provisional specialist for a recurring need (opt-in setting).",
            aliases=("new specialist", "create expert", "make delegation agent"),
        )
        create_tool.delegation_runtime = runtime  # type: ignore[attr-defined]
        tools_to_add.append(create_tool)
    for tool in tools_to_add:
        if not registry.has(tool.name):
            registry.add(tool)
            added.append(tool)
    return added


def wire_specialist_runner(registry: ToolRegistry, runner: Any) -> bool:
    """Set the runner on already-registered specialist tools."""

    found = False
    for tool in registry.all():
        runtime = getattr(tool, "delegation_runtime", None)
        if runtime is not None and hasattr(runtime, "set_runner"):
            runtime.set_runner(runner)
            found = True
    return found
