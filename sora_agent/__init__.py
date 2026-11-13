"""SoraAgent: model-agnostic agent harness with approval and quality gates."""

from .agent import Agent, AgentResult
from .calendar import (
    IMPORTANCE_LEVELS,
    CalendarItem,
    CalendarStore,
    default_calendar,
    register_calendar_tools,
)
from .clock import CLOCK_HEADING, ClockProvider, clock_section, wrap_clock
from .identity import IdentityConfig, IdentityStore, default_identity
from .identity_provider import IdentityProvider
from .providers import MockProvider, ModelTurn, ToolCall
from .runner import Runner, RunResult, RunSpec, Step
from .state_machine import POStateMachine
from .tool_search import (
    FIND_TOOLS_NAME,
    Candidate,
    FindToolsTool,
    LexicalRanker,
    ToolGate,
    ToolRanker,
)
from .toolpack import CORE_TOOL_NAMES, ToolPackConfig, register_core_tools
from .tools import ToolExecutor, ToolRegistry, ToolResult

__all__ = [
    "CLOCK_HEADING",
    "CORE_TOOL_NAMES",
    "FIND_TOOLS_NAME",
    "IMPORTANCE_LEVELS",
    "Agent",
    "AgentResult",
    "CalendarItem",
    "CalendarStore",
    "Candidate",
    "ClockProvider",
    "FindToolsTool",
    "IdentityConfig",
    "IdentityProvider",
    "IdentityStore",
    "LexicalRanker",
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
    "ToolPackConfig",
    "ToolRanker",
    "ToolRegistry",
    "ToolResult",
    "clock_section",
    "default_calendar",
    "default_identity",
    "register_calendar_tools",
    "register_core_tools",
    "wrap_clock",
]
__version__ = "0.1.3"
