"""SoraAgent: model-agnostic agent harness with approval and quality gates."""

from .agent import Agent, AgentResult
from .identity import IdentityConfig, IdentityStore, default_identity
from .identity_provider import IdentityProvider
from .providers import MockProvider, ModelTurn, ToolCall
from .runner import Runner, RunResult, RunSpec, Step
from .state_machine import POStateMachine
from .tool_search import FIND_TOOLS_NAME, FindToolsTool, ToolGate
from .tools import ToolExecutor, ToolRegistry, ToolResult

__all__ = [
    "FIND_TOOLS_NAME",
    "Agent",
    "AgentResult",
    "FindToolsTool",
    "IdentityConfig",
    "IdentityProvider",
    "IdentityStore",
    "MockProvider",
    "ModelTurn",
    "POStateMachine",
    "RunResult",
    "RunSpec",
    "Runner",
    "Step",
    "ToolCall",
    "ToolExecutor",
    "ToolGate",
    "ToolRegistry",
    "ToolResult",
    "default_identity",
]
__version__ = "0.1.0"
