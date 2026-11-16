"""Hermes-style core tool pack for SoraAgent.

All tools in this pack are ordinary hidden registry entries.  ``find_tools`` is
the only way to reach them: the initial model tool list remains
``find_tools`` plus ``memory_recall`` (when memory is enabled), and the required
first-action policy is untouched.

Tools that can change the workspace, spend money, or execute code are marked
``needs_approval=True`` so SoraAgent's existing approval gate stays in front of
them.
"""

from __future__ import annotations

from typing import Any

from ..delegation import DelegationStore, register_specialist_tools
from ..tools import Tool, ToolRegistry
from .config import ToolPackConfig
from .delegation import DelegateRunner, register_delegation_tools
from .execution import register_execution_tools
from .files import register_file_tools
from .planning import ClarificationGate, register_planning_tools
from .shell import register_shell_tools
from .web import register_web_tools

CORE_TOOL_NAMES = (
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "terminal",
    "process_manage",
    "web_search",
    "web_extract",
    "todo_list",
    "clarify",
    "execute_code",
    "delegate_task",
    "find_delegates",
    "delegate_to_agent",
)

__all__ = [
    "CORE_TOOL_NAMES",
    "ToolPackConfig",
    "register_core_tools",
]


def register_core_tools(
    registry: ToolRegistry,
    config: ToolPackConfig | None = None,
    *,
    clarify_gate: ClarificationGate | None = None,
    delegate_runner: DelegateRunner | None = None,
    delegation_store: DelegationStore | None = None,
    specialist_runner: Any = None,
    allow_create: bool = False,
    max_agents: int = 20,
    max_per_day: int = 3,
    enable_specialists: bool = True,
) -> list[Tool]:
    """Register the core tool pack on ``registry`` and return what was added.

    Existing tools with the same names are left untouched.  The tools remain
    hidden to the model until ``find_tools`` loads them.
    """

    pack = config or ToolPackConfig.from_env()
    added: list[Tool] = []
    added.extend(register_file_tools(registry, pack))
    added.extend(register_shell_tools(registry, pack))
    added.extend(register_web_tools(registry, pack))
    added.extend(register_planning_tools(registry, pack, clarify_gate=clarify_gate))
    added.extend(register_execution_tools(registry, pack))
    added.extend(
        register_delegation_tools(registry, pack, delegate_runner=delegate_runner)
    )
    if enable_specialists:
        added.extend(
            register_specialist_tools(
                registry,
                delegation_store or DelegationStore(),
                runner=specialist_runner,
                allow_create=allow_create,
                max_agents=max_agents,
                max_per_day=max_per_day,
            )
        )
    return added
