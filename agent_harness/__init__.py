"""ai-agent-scaffold: model-agnostic agent harness with approval and quality gates."""

from .agent import Agent, AgentResult
from .providers import MockProvider
from .state_machine import POStateMachine

__all__ = ["Agent", "AgentResult", "MockProvider", "POStateMachine"]
__version__ = "0.1.0"
