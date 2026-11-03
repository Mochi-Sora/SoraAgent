"""Shared per-run orchestration for the agent facade and channel handlers.

The ``Runner`` stays a pure model/tool loop.  This module owns the per-run
assembly around it: registry isolation, lazy tool discovery, memory-tool
registration, the quality guard, and the executor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .eval import QualityGate
from .providers import Message
from .runner import RunEvent, Runner, RunResult, RunSpec
from .tool_search import FIND_TOOLS_NAME, FindToolsTool, ToolGate
from .tools import ToolCall, ToolExecutor, ToolRegistry, ToolResult

log = logging.getLogger(__name__)


def wrap_identity(identity: Any, provider: Any) -> Any:
    """Wrap ``provider`` with identity support, failing open."""
    if identity is None or getattr(provider, "__sora_identity__", False):
        return provider
    try:
        return identity.wrap_provider(provider)
    except Exception as exc:  # noqa: BLE001 - identity is optional
        log.warning("identity provider wrapper failed: %s", exc)
        return provider


@dataclass
class RunRequest:
    """One isolated agent run."""

    provider: Any
    registry: ToolRegistry
    messages: list[Message]
    memory_tool: Any = None
    approval: Any = None
    quality: QualityGate | None = None
    max_steps: int = 10
    max_result_chars: int = 4000
    tool_timeout: float = 30.0
    context: str = ""
    after_execute: Callable[[ToolCall, ToolResult], None] | None = None
    required_first_tool: str | None = None
    on_delta: Callable[[str, int], None] | None = None
    should_stop: Callable[[], bool] | None = None
    on_event: Callable[[RunEvent], None] | None = None
    provider_retries: int = 2
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0


class AgentRuntime:
    """Build one isolated run around the pure ``Runner``."""

    @staticmethod
    def run(request: RunRequest) -> RunResult:
        if request.registry.has(FIND_TOOLS_NAME):
            raise ValueError(
                f"the tool name {FIND_TOOLS_NAME!r} is reserved for lazy "
                "tool discovery"
            )

        run_registry = request.registry.copy()
        memory_name = str(getattr(request.memory_tool, "name", "") or "")
        always_visible = {FIND_TOOLS_NAME}
        if request.memory_tool is not None and memory_name:
            run_registry.add(request.memory_tool)
            always_visible.add(memory_name)

        gate = ToolGate(run_registry, always_visible=always_visible)
        run_registry.add(FindToolsTool(gate))

        required = str(request.required_first_tool or "").strip()
        if not required and memory_name and run_registry.has(memory_name):
            required = memory_name

        executor = ToolExecutor(
            registry=run_registry,
            approval=request.approval,
            max_result_chars=request.max_result_chars,
            after_execute=request.after_execute,
            tool_timeout=request.tool_timeout,
        )

        guard: Callable[[str, str], tuple[bool, Any]] | None = None
        if request.quality is not None:
            quality = request.quality

            def _guard(proposal: str, context: str) -> tuple[bool, Any]:
                report = quality.grade(proposal, context)
                return bool(getattr(report, "passed", False)), report

            guard = _guard

        return Runner(request.provider, executor).run(
            RunSpec(
                messages=request.messages,
                tools=gate.visible(),
                tool_provider=gate.visible,
                required_first_tool=required or None,
                max_steps=request.max_steps,
                context=request.context,
                guard=guard,
                on_delta=request.on_delta,
                should_stop=request.should_stop,
                on_event=request.on_event,
                provider_retries=request.provider_retries,
                retry_base_delay=request.retry_base_delay,
                retry_max_delay=request.retry_max_delay,
            )
        )
