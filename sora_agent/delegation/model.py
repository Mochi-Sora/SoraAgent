"""Persistent specialist definitions (delegation agents)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

FORBIDDEN_TOOLS = frozenset(
    {
        "find_delegates",
        "delegate_to_agent",
        "create_delegation_agent",
        "find_tools",
    }
)

TOOL_PACKS: dict[str, tuple[str, ...]] = {
    "research": (
        "web_search",
        "web_extract",
        "read_file",
        "search_files",
        "memory_recall",
    ),
    "files": ("read_file", "write_file", "patch", "search_files"),
    "code": ("execute_code", "terminal", "process_manage"),
    "planning": ("todo_list", "clarify"),
    "web": ("web_search", "web_extract"),
}


class DelegationError(ValueError):
    """A specialist manifest is invalid."""


@dataclass
class DelegationAgent:
    id: str
    name: str
    description: str = ""
    when_to_use: str = ""
    prompt: str = ""
    provider: str = ""
    model: str = ""
    tools: tuple[str, ...] = ("research",)
    memory: str = "read"
    approval: str = "ask"
    max_steps: int = 8
    max_seconds: float = 300.0
    status: str = "active"
    created_by: str = "user"
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not SLUG_RE.fullmatch(self.id or ""):
            raise DelegationError(
                "agent id must be a lowercase slug (letters, digits, - or _)"
            )
        if not self.name.strip():
            raise DelegationError("agent name is required")
        if not self.prompt.strip():
            raise DelegationError("agent prompt is required")
        if not self.description.strip() and not self.when_to_use.strip():
            raise DelegationError("description or when_to_use is required")
        if self.memory not in {"read", "none"}:
            raise DelegationError("memory must be 'read' or 'none'")
        if self.approval not in {"ask", "off"}:
            raise DelegationError("approval must be 'ask' or 'off'")
        if self.status not in {"active", "paused"}:
            raise DelegationError("status must be 'active' or 'paused'")
        if not 1 <= int(self.max_steps) <= 50:
            raise DelegationError("max_steps must be between 1 and 50")
        if not 10 <= float(self.max_seconds) <= 3600:
            raise DelegationError("max_seconds must be between 10 and 3600")
        selected = self.resolved_tools()
        forbidden = sorted(FORBIDDEN_TOOLS.intersection(selected))
        if forbidden:
            raise DelegationError(
                "specialists cannot use delegation control tools: "
                + ", ".join(forbidden)
            )
        if not selected:
            raise DelegationError("at least one tool grant is required")
        if not self.provider.strip():
            self.provider = ""
        if not self.model.strip():
            self.model = ""

    def resolved_tools(self) -> tuple[str, ...]:
        """Expand packs into concrete tool names, preserving order."""

        values: list[str] = []
        for item in self.tools:
            name = str(item or "").strip()
            if not name:
                continue
            values.extend(TOOL_PACKS.get(name, (name,)))
        seen: set[str] = set()
        result: list[str] = []
        for name in values:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return tuple(result)

    def routing_text(self) -> str:
        return " ".join(
            part for part in (self.name, self.description, self.when_to_use) if part
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "prompt": self.prompt,
            "provider": self.provider,
            "model": self.model,
            "tools": list(self.tools),
            "memory": self.memory,
            "approval": self.approval,
            "max_steps": int(self.max_steps),
            "max_seconds": float(self.max_seconds),
            "status": self.status,
            "created_by": self.created_by,
            "version": int(self.version),
            **({"metadata": self.metadata} if self.metadata else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelegationAgent:
        if not isinstance(data, dict):
            raise DelegationError("agent manifest must be a JSON object")
        raw_tools = data.get("tools", [])
        if isinstance(raw_tools, str):
            raw_tools = [item for item in raw_tools.split(",")]
        agent = cls(
            id=str(data.get("id", "")).strip(),
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description", "")).strip(),
            when_to_use=str(data.get("when_to_use", "")).strip(),
            prompt=str(data.get("prompt", "")).strip(),
            provider=str(data.get("provider", "")).strip(),
            model=str(data.get("model", "")).strip(),
            tools=tuple(
                str(item).strip()
                for item in raw_tools
                if str(item).strip()
            ),
            memory=str(data.get("memory", "read") or "read"),
            approval=str(data.get("approval", "ask") or "ask"),
            max_steps=int(data.get("max_steps", 8) or 8),
            max_seconds=float(data.get("max_seconds", 300) or 300),
            status=str(data.get("status", "active") or "active"),
            created_by=str(data.get("created_by", "user") or "user"),
            version=int(data.get("version", 1) or 1),
            metadata=dict(data.get("metadata") or {}),
        )
        agent.validate()
        return agent
