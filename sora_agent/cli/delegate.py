"""CLI for user-created specialists (delegation agents)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..delegation import DelegationError, DelegationStore, default_root
from . import ui

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Create and manage persistent specialist agents.",
)


def _store() -> DelegationStore:
    return DelegationStore()


@app.command("list")
def list_agents() -> None:
    """List specialists, including paused ones."""
    agents = _store().list(include_paused=True)
    if not agents:
        ui.warning("No specialists yet. Create one with `sora delegate create`.")
        ui.hint(f"Store: {default_root()}")
        return
    ui.grid_table(
        "Specialists",
        ["id", "name", "status", "model", "tools"],
        [
            (
                agent.id,
                agent.name,
                agent.status,
                f"{agent.provider}/{agent.model}".strip("/") or "main",
                ", ".join(agent.tools),
            )
            for agent in agents
        ],
    )
    ui.hint(f"Store: {default_root()}")


@app.command("show")
def show(agent_id: str = typer.Argument(..., help="Specialist id")) -> None:
    """Print one specialist manifest."""
    agent = _store().get(agent_id)
    if agent is None:
        ui.error(f"No specialist with id {agent_id!r}.")
        raise typer.Exit(1)
    typer.echo(json.dumps(agent.to_dict(), indent=2, ensure_ascii=False))


@app.command("create")
def create(
    name: str = typer.Option(..., "--name", "-n", help="Display name"),
    description: str = typer.Option("", "--description", "-d", help="One-line purpose"),
    when: str = typer.Option("", "--when", "-w", help="When the orchestrator should use it"),
    prompt: str = typer.Option("", "--prompt", "-p", help="Specialist system prompt"),
    prompt_file: str = typer.Option(
        "", "--prompt-file", help="Read the system prompt from a file"
    ),
    agent_id: str = typer.Option("", "--id", help="Stable slug (defaults to name)"),
    tools: str = typer.Option(
        "research", "--tools", "-t", help="Comma-separated packs or tool names"
    ),
    provider: str = typer.Option("", "--provider", help="Provider override"),
    model: str = typer.Option("", "--model", "-m", help="Model override"),
    memory: str = typer.Option("read", "--memory", help="read or none"),
    approval: str = typer.Option("ask", "--approval", help="ask or off"),
    max_steps: int = typer.Option(8, "--max-steps", help="Tool steps per run"),
    max_seconds: float = typer.Option(300.0, "--max-seconds", help="Wall-clock budget"),
) -> None:
    """Create a persistent specialist."""
    if prompt_file:
        path = Path(prompt_file).expanduser()
        try:
            prompt = path.read_text(encoding="utf-8")
        except OSError as exc:
            ui.error(f"Could not read {path}: {exc}")
            raise typer.Exit(1) from exc
    if not prompt.strip():
        ui.error("Provide --prompt or --prompt-file.")
        raise typer.Exit(1)
    selected = tuple(item.strip() for item in tools.split(",") if item.strip())
    try:
        agent = _store().create(
            name=name,
            description=description,
            when_to_use=when,
            prompt=prompt,
            tools=selected,
            provider=provider,
            model=model,
            memory=memory,
            approval=approval,
            max_steps=max_steps,
            max_seconds=max_seconds,
            agent_id=agent_id or None,
        )
    except DelegationError as exc:
        ui.error(str(exc))
        raise typer.Exit(1) from exc
    ui.success(f"Created specialist {agent.id!r} ({agent.name}).")
    ui.hint(f"Ask the main agent to `find_delegates` for tasks matching: {agent.when_to_use or agent.description}")


@app.command("delete")
def delete(
    agent_id: str = typer.Argument(..., help="Specialist id"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    workspace: bool = typer.Option(
        False, "--workspace", help="Also delete the specialist workspace"
    ),
) -> None:
    """Delete a specialist manifest."""
    if not yes and not typer.confirm(f"Delete specialist {agent_id!r}?"):
        raise typer.Exit(0)
    if not _store().delete(agent_id, remove_workspace=workspace):
        ui.error(f"No specialist with id {agent_id!r}.")
        raise typer.Exit(1)
    ui.success(f"Deleted specialist {agent_id!r}.")


@app.command("promote")
def promote(agent_id: str = typer.Argument(...)) -> None:
    """Clear the provisional flag on a specialist created by the main agent."""
    store = _store()
    agent = store.get(agent_id)
    if agent is None:
        ui.error(f"No specialist with id {agent_id!r}.")
        raise typer.Exit(1)
    agent.metadata.pop("provisional", None)
    store.save(agent)
    ui.success(f"Specialist {agent.id!r} is now a regular specialist.")


@app.command("pause")
def pause(agent_id: str = typer.Argument(...)) -> None:
    """Pause a specialist without deleting it."""
    _set_status(agent_id, "paused")


@app.command("resume")
def resume(agent_id: str = typer.Argument(...)) -> None:
    """Resume a paused specialist."""
    _set_status(agent_id, "active")


def _set_status(agent_id: str, status: str) -> None:
    store = _store()
    agent = store.get(agent_id)
    if agent is None:
        ui.error(f"No specialist with id {agent_id!r}.")
        raise typer.Exit(1)
    agent.status = status
    store.save(agent)
    ui.success(f"Specialist {agent.id!r} is now {status}.")
