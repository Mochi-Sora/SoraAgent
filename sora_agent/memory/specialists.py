"""Discover specialist daily stores for dreaming/consolidation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .daily import DailyStore

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SpecialistSource:
    agent_id: str
    store: DailyStore
    root: Path
    name: str = ""


def specialists_root() -> Path:
    configured = os.getenv("SORA_DELEGATIONS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".sora_agent" / "delegations"


def discover_specialists(root: Path | None = None) -> list[SpecialistSource]:
    base = Path(root).expanduser() if root is not None else specialists_root()
    if not base.exists():
        return []
    sources: list[SpecialistSource] = []
    for agent_dir in sorted(base.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_id = agent_dir.name
        if not SLUG_RE.fullmatch(agent_id):
            continue
        daily_dir = agent_dir / "workspace" / "daily"
        if not daily_dir.exists():
            continue
        if not any(daily_dir.glob("*.db")):
            continue
        store = DailyStore(daily_dir, session_id=f"specialist:{agent_id}")
        name = agent_id
        manifest = agent_dir / "agent.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(data, dict) and str(data.get("name", "")).strip():
                    name = str(data["name"]).strip()
            except (OSError, ValueError):
                pass
        sources.append(
            SpecialistSource(
                agent_id=agent_id, store=store, root=daily_dir, name=name
            )
        )
    return sources


def has_pending(sources: list[SpecialistSource], now) -> bool:
    return any(source.store.pending_days(now) for source in sources)
