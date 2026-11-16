"""User-created persistent specialists, separate from ephemeral sub-agents."""

from .factory import SpecialistAgentRunner, build_specialist_registry
from .model import DelegationAgent, DelegationError
from .router import best, rank
from .runtime import DelegationRuntime, SpecialistRunner
from .store import DelegationStore, default_root
from .tools import register_specialist_tools, wire_specialist_runner

__all__ = [
    "DelegationAgent",
    "DelegationError",
    "DelegationRuntime",
    "DelegationStore",
    "SpecialistAgentRunner",
    "SpecialistRunner",
    "best",
    "build_specialist_registry",
    "default_root",
    "rank",
    "register_specialist_tools",
    "wire_specialist_runner",
]
