"""Filesystem-backed specialist registry."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from .model import SLUG_RE, DelegationAgent, DelegationError

log = logging.getLogger(__name__)


def default_root() -> Path:
    configured = os.getenv("SORA_DELEGATIONS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".sora_agent" / "delegations"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug[:64] or "specialist"


class DelegationStore:
    """Read and write ``<root>/<id>/agent.json`` manifests."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root else default_root()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def path_for(self, agent_id: str) -> Path:
        return self.root / agent_id / "agent.json"

    def list(self, *, include_paused: bool = False) -> list[DelegationAgent]:
        if not self.root.exists():
            return []
        agents: list[DelegationAgent] = []
        for manifest in sorted(self.root.glob("*/agent.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                agent = DelegationAgent.from_dict(data)
            except (OSError, ValueError, TypeError, DelegationError) as exc:
                log.warning("ignoring invalid specialist manifest %s: %s", manifest, exc)
                continue
            if include_paused or agent.status == "active":
                agents.append(agent)
        return agents

    def get(self, agent_id: str) -> DelegationAgent | None:
        if not SLUG_RE.fullmatch(str(agent_id or "")):
            return None
        path = self.path_for(agent_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return DelegationAgent.from_dict(data)
        except (OSError, ValueError, TypeError, DelegationError) as exc:
            log.warning("could not read specialist %s: %s", agent_id, exc)
            return None

    def save(self, agent: DelegationAgent) -> DelegationAgent:
        agent.validate()
        self.ensure()
        directory = self.root / agent.id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        target = directory / "agent.json"
        payload = json.dumps(agent.to_dict(), indent=2, ensure_ascii=False) + "\n"
        handle, temporary = tempfile.mkstemp(
            prefix=".agent-", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        (directory / "workspace").mkdir(exist_ok=True)
        return agent

    def create(
        self,
        *,
        name: str,
        description: str = "",
        when_to_use: str = "",
        prompt: str,
        tools: tuple[str, ...] = ("research",),
        provider: str = "",
        model: str = "",
        memory: str = "read",
        approval: str = "ask",
        max_steps: int = 8,
        max_seconds: float = 300.0,
        agent_id: str | None = None,
        created_by: str = "user",
    ) -> DelegationAgent:
        slug = slugify(agent_id or name)
        if self.get(slug) is not None:
            raise DelegationError(f"a specialist with id {slug!r} already exists")
        agent = DelegationAgent(
            id=slug,
            name=name,
            description=description,
            when_to_use=when_to_use,
            prompt=prompt,
            provider=provider,
            model=model,
            tools=tuple(tools),
            memory=memory,
            approval=approval,
            max_steps=max_steps,
            max_seconds=max_seconds,
            created_by=created_by,
        )
        return self.save(agent)

    def delete(self, agent_id: str, *, remove_workspace: bool = False) -> bool:
        directory = self.root / agent_id
        manifest = directory / "agent.json"
        if not manifest.exists():
            return False
        manifest.unlink()
        if remove_workspace:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
        return True

    def workspace_for(self, agent: DelegationAgent) -> Path:
        directory = self.root / agent.id / "workspace"
        directory.mkdir(parents=True, exist_ok=True)
        return directory
