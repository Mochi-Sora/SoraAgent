"""SoraAgent: model-agnostic agent harness with approval and quality gates."""

from .agent import Agent, AgentResult
from .identity import IdentityConfig, IdentityStore, default_identity
from .identity_provider import IdentityProvider
from .providers import MockProvider
from .state_machine import POStateMachine

__all__ = [
    "Agent",
    "AgentResult",
    "IdentityConfig",
    "IdentityProvider",
    "IdentityStore",
    "MockProvider",
    "POStateMachine",
    "default_identity",
]
__version__ = "0.1.0"
