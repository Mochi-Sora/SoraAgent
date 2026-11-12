"""Tests for the optional per-tool idempotency guard."""

from __future__ import annotations

from sora_agent.approval import AutoApprovalGate
from sora_agent.providers import ToolCall
from sora_agent.tools import Tool, ToolExecutor, ToolRegistry


def _registry(counter: list[int], *, dedupe: bool) -> ToolRegistry:
    registry = ToolRegistry()

    def send(message: str) -> str:
        counter.append(1)
        return "sent:" + message

    registry.add(
        Tool(
            send,
            "send_message",
            "Send a message.",
            needs_approval=True,
            dedupe=dedupe,
        )
    )
    return registry


def test_dedupe_tool_returns_cached_result_without_second_side_effect():
    counter: list[int] = []
    executor = ToolExecutor(
        _registry(counter, dedupe=True), approval=AutoApprovalGate()
    )
    call = ToolCall(id="1", name="send_message", arguments={"message": "hi"})

    first = executor.execute(call)
    second = executor.execute(call)

    assert first.executed is True
    assert second.executed is False
    assert "deduplicated" in second.content
    assert len(counter) == 1


def test_non_dedupe_tool_executes_every_call():
    counter: list[int] = []
    executor = ToolExecutor(
        _registry(counter, dedupe=False), approval=AutoApprovalGate()
    )
    call = ToolCall(id="1", name="send_message", arguments={"message": "hi"})

    executor.execute(call)
    executor.execute(call)

    assert len(counter) == 2
