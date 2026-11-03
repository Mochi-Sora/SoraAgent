"""Tests for run metrics, events, token usage, and /status."""

from __future__ import annotations

from types import SimpleNamespace

from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.types import InboundMessage
from sora_agent.providers import (
    ModelTurn,
    OpenAIChatProvider,
    TokenUsage,
    ToolCall,
)
from sora_agent.runner import Runner, RunSpec
from sora_agent.tools import Tool, ToolExecutor, ToolRegistry


class TwoStepProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(
                tool_calls=[ToolCall("c1", "echo", {"text": "hi"})],
                usage=TokenUsage(10, 5, 15),
            )
        return ModelTurn(text="done", usage=TokenUsage(20, 7, 27))


class BoomProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(tool_calls=[ToolCall("c1", "boom", {})])
        return ModelTurn(text="done")


class FlakyProvider:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return ModelTurn(text="done")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(Tool(lambda text: text, "echo", "Echo text."))
    return registry


def _run(provider, registry=None, **overrides):
    registry = registry or _registry()
    spec = {
        "messages": [],
        "tools": registry.all(),
        "max_steps": 3,
        "provider_retries": 0,
    }
    spec.update(overrides)
    return Runner(provider, ToolExecutor(registry)).run(RunSpec(**spec))


def _inbound(text: str) -> InboundMessage:
    return InboundMessage("fake", "7", "42", text)


def test_metrics_count_steps_tools_and_tokens():
    result = _run(TwoStepProvider())

    metrics = result.metrics
    assert result.stop_reason == "completed"
    assert metrics.steps == 2
    assert metrics.provider_calls == 2
    assert metrics.provider_retries == 0
    assert metrics.tool_calls == 1
    assert metrics.tool_errors == 0
    assert metrics.prompt_tokens == 30
    assert metrics.completion_tokens == 12
    assert metrics.total_tokens == 42
    assert metrics.duration >= 0
    assert metrics.tool_duration >= 0
    assert metrics.provider_duration >= 0


def test_repeated_calls_are_counted_as_redundant():
    class RepeatProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete_turn(self, messages, tools) -> ModelTurn:
            self.calls += 1
            if self.calls <= 2:
                return ModelTurn(
                    tool_calls=[
                        ToolCall(
                            f"c{self.calls}", "echo", {"text": "hi"}
                        )
                    ]
                )
            return ModelTurn(text="done")

    result = _run(RepeatProvider())

    assert result.metrics.tool_calls == 2
    assert result.metrics.executed_tool_calls == 2
    assert result.metrics.redundant_tool_calls == 1


def test_tool_errors_are_counted():
    registry = ToolRegistry()

    def boom() -> str:
        raise RuntimeError("broken")

    registry.add(Tool(boom, "boom", "Fail."))
    result = _run(BoomProvider(), registry)

    assert result.metrics.tool_calls == 1
    assert result.metrics.tool_errors == 1


def test_events_describe_the_run():
    events = []
    result = _run(TwoStepProvider(), on_event=events.append)

    kinds = [event.kind for event in events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    assert kinds.count("provider_finished") == 2
    tool_event = next(event for event in events if event.kind == "tool_finished")
    assert tool_event.name == "echo"
    assert tool_event.step == 0
    assert tool_event.ok is True
    assert tool_event.duration >= 0
    assert events[-1].detail == result.stop_reason


def test_retries_are_visible_in_metrics_and_events():
    events = []
    result = _run(
        FlakyProvider(failures=1),
        provider_retries=2,
        retry_base_delay=0.001,
        retry_max_delay=0.001,
        on_event=events.append,
    )

    assert result.stop_reason == "completed"
    assert result.metrics.provider_calls == 2
    assert result.metrics.provider_retries == 1
    assert any(event.kind == "provider_retry" for event in events)


def test_a_broken_observer_cannot_break_the_run():
    def broken(event) -> None:
        raise RuntimeError("observer is down")

    result = _run(TwoStepProvider(), on_event=broken)
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# provider usage
# ---------------------------------------------------------------------------


def test_non_streaming_usage_is_parsed(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hi", tool_calls=[])
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3, completion_tokens=4, total_tokens=7
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    provider = OpenAIChatProvider(api_key="k")
    monkeypatch.setattr(provider, "_client", lambda: client)

    turn = provider.complete_turn([], [])

    assert turn.text == "hi"
    assert turn.usage == TokenUsage(3, 4, 7)


def test_streaming_usage_is_parsed(monkeypatch):
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="he", tool_calls=None)
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="llo", tool_calls=None)
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=5, completion_tokens=2, total_tokens=7
            ),
        ),
    ]
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return iter(chunks)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = OpenAIChatProvider(api_key="k")
    monkeypatch.setattr(provider, "_client", lambda: client)

    turn = provider.stream_turn([], [], lambda chunk: None)

    assert turn.text == "hello"
    assert turn.usage == TokenUsage(5, 2, 7)
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


def test_status_reports_the_last_run():
    class PlainProvider:
        def complete_turn(self, messages, tools) -> ModelTurn:
            return ModelTurn(text="hi")

    handler = ChatHandler(PlainProvider(), max_steps=1)
    handler.handle(_inbound("hi"))

    status = handler.handle_command(_inbound("/status"))

    assert status is not None
    assert "Last run: completed" in status
    assert "1 steps" in status
    assert "0 tool calls" in status
    assert "token usage unavailable" in status
    assert "ASR 0/0" in status
    assert "eval pass" in status


def test_status_without_a_run():
    class PlainProvider:
        def complete_turn(self, messages, tools) -> ModelTurn:
            return ModelTurn(text="hi")

    handler = ChatHandler(PlainProvider(), max_steps=1)
    assert handler.handle_command(_inbound("/status")) == (
        "No run in this session yet."
    )
