"""Tests for provider limits, retries, and process-isolated tools."""

from __future__ import annotations

import json
import os
import sys
import threading
from types import SimpleNamespace

from sora_agent.isolation import block_forever
from sora_agent.providers import (
    ModelTurn,
    OpenAIChatProvider,
    ToolCall,
)
from sora_agent.runner import Runner, RunSpec
from sora_agent.tools import Tool, ToolExecutor, ToolRegistry


class FlakyProvider:
    def __init__(self, failures: int, text: str = "done") -> None:
        self.failures = failures
        self.text = text
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return ModelTurn(text=self.text)


def _run(provider, **overrides):
    spec = {
        "messages": [],
        "tools": [],
        "max_steps": 2,
        "provider_retries": 2,
        "retry_base_delay": 0.001,
        "retry_max_delay": 0.002,
    }
    spec.update(overrides)
    return Runner(provider, ToolExecutor(ToolRegistry())).run(RunSpec(**spec))


# ---------------------------------------------------------------------------
# provider timeout and retries
# ---------------------------------------------------------------------------


def test_openai_client_gets_a_bounded_timeout(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI)
    )
    provider = OpenAIChatProvider(api_key="k", timeout=30.0)
    provider._client()

    assert captured["timeout"] == 30.0


def test_transient_provider_failure_is_retried():
    provider = FlakyProvider(failures=2)

    result = _run(provider)

    assert result.stop_reason == "completed"
    assert provider.calls == 3
    assert result.final_answer == "done"


def test_retries_stop_after_the_budget():
    provider = FlakyProvider(failures=99)

    result = _run(provider, provider_retries=1)

    assert result.stop_reason == "provider_error"
    assert provider.calls == 2
    assert "transient" in result.error


def test_cancellation_interrupts_the_backoff():
    provider = FlakyProvider(failures=99)
    state = {"stop": False}
    timer = threading.Timer(0.05, lambda: state.__setitem__("stop", True))
    timer.start()
    try:
        result = _run(
            provider,
            provider_retries=5,
            retry_base_delay=0.5,
            retry_max_delay=0.5,
            should_stop=lambda: state["stop"],
        )
    finally:
        timer.cancel()

    assert result.stop_reason == "cancelled"
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# process-isolated tools
# ---------------------------------------------------------------------------


def test_isolated_tool_runs_in_another_process():
    registry = ToolRegistry()
    registry.add(Tool(os.getpid, "pid", "Return the process id.", isolated=True))
    executor = ToolExecutor(registry, tool_timeout=10.0)

    result = executor.execute(ToolCall("c1", "pid", {}))

    assert result.is_error is False
    assert result.executed is True
    assert result.content.isdigit()
    assert int(result.content) != os.getpid()


def test_isolated_tool_is_killed_on_timeout():
    registry = ToolRegistry()
    registry.add(
        Tool(block_forever, "hang", "Never returns.", isolated=True)
    )
    executor = ToolExecutor(registry, tool_timeout=0.5)

    result = executor.execute(ToolCall("c1", "hang", {}))

    assert result.is_error is True
    assert "exceeded" in result.content
    assert result.executed is False


def test_unpicklable_isolated_tool_fails_cleanly():
    registry = ToolRegistry()
    registry.add(
        Tool(lambda: "nope", "closure", "Not picklable.", isolated=True)
    )
    executor = ToolExecutor(registry, tool_timeout=2.0)

    result = executor.execute(ToolCall("c1", "closure", {}))

    assert result.is_error is True
    assert "picklable" in result.content


def test_isolated_tool_reports_child_errors():
    registry = ToolRegistry()
    registry.add(
        Tool(json.loads, "parse", "Parse JSON.", isolated=True)
    )

    result = ToolExecutor(registry).execute(
        ToolCall("c1", "parse", {"s": "{"})
    )

    assert result.is_error is True
    assert "JSONDecodeError" in result.content
