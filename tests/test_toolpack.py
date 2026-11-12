"""Tests for the built-in Hermes-style core tool pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sora_agent.agent import Agent
from sora_agent.providers import Message, ModelTurn
from sora_agent.runtime_setup import build_registry
from sora_agent.settings import advanced_settings, recommended_settings
from sora_agent.tool_search import FindToolsTool, ToolGate
from sora_agent.toolpack import CORE_TOOL_NAMES, ToolPackConfig, register_core_tools
from sora_agent.tools import ToolRegistry


class RecordingProvider:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or ["done"])
        self.calls: list[list[Message]] = []
        self.tools: list[list[str]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        self.tools.append([tool.name for tool in tools])
        text = self.replies.pop(0) if self.replies else "done"
        return ModelTurn(text=text)


class FakeRecallTool:
    name = "memory_recall"
    description = "Read relevant memory as untrusted evidence."
    needs_approval = False

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }

    def call(self, query: str = "") -> str:
        return "Memory results: no relevant memory found."


class FakeMemory:
    def tool(self) -> FakeRecallTool:
        return FakeRecallTool()

    def before_run(self, task: str, state: Any = None) -> None:
        return None

    def after_run(self, result: Any, state: Any = None) -> None:
        return None


def _registry(tmp_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    register_core_tools(registry, ToolPackConfig(root=tmp_path))
    return registry


def test_core_pack_names_and_hidden_initial_list(tmp_path):
    registry = _registry(tmp_path)
    assert [tool.name for tool in registry.all()] == list(CORE_TOOL_NAMES)

    provider = RecordingProvider()
    Agent(
        provider=provider,
        registry=registry,
        memory=False,
        identity=False,
        clock=False,
        calendar=False,
    ).run("hello")

    assert provider.tools[0] == ["find_tools"]
    assert all(name not in provider.tools[0] for name in CORE_TOOL_NAMES)


def test_recommended_settings_register_pack_advanced_does_not(monkeypatch):
    monkeypatch.delenv("SORA_TOOLPACK", raising=False)
    recommended = build_registry(recommended_settings())
    assert [tool.name for tool in recommended.all()] == list(CORE_TOOL_NAMES)
    assert build_registry(advanced_settings()).all() == []

    monkeypatch.setenv("SORA_TOOLPACK", "0")
    assert build_registry(recommended_settings()).all() == []


def test_memory_recall_still_must_be_the_first_action(tmp_path):
    provider = RecordingProvider(["direct answer", "still direct", "direct"])
    result = Agent(
        provider=provider,
        registry=_registry(tmp_path),
        memory=FakeMemory(),
        identity=False,
        clock=False,
        calendar=False,
        max_steps=5,
    ).run("Check something")

    assert result.stop_reason == "memory_recall_required"
    assert set(provider.tools[0]) == {"memory_recall", "find_tools"}
    assert not any(name in provider.tools[0] for name in CORE_TOOL_NAMES)


def test_file_tools_round_trip_and_search(tmp_path):
    registry = _registry(tmp_path)
    write = registry.get("write_file")
    read = registry.get("read_file")
    patch = registry.get("patch")
    search = registry.get("search_files")

    assert "wrote" in write.call(path="notes/a.txt", content="hello\nworld\n")
    text = read.call(path="notes/a.txt")
    assert "1|hello" in text
    assert "patched" in patch.call(
        path="notes/a.txt", old_text="world", new_text="there"
    )
    assert "there" in read.call(path="notes/a.txt")
    assert "a.txt:2" in search.call(pattern="there", path="notes")
    assert "a.txt" in search.call(pattern="*.txt", path="notes", target="files")
    assert write.needs_approval is True
    assert patch.needs_approval is True


def test_paths_are_confined_to_workspace(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ValueError):
        registry.get("read_file").call(path=str(tmp_path.parent / "secret"))


def test_terminal_and_background_process(tmp_path):
    registry = _registry(tmp_path)
    output = registry.get("terminal").call(command="printf hello")
    assert "exit_code: 0" in output
    assert "hello" in output

    started = registry.get("terminal").call(
        command="printf 'a\\n'; sleep 0.4; printf 'b\\n'",
        background=True,
    )
    session_id = next(
        line.split(":", 1)[1].strip()
        for line in started.splitlines()
        if line.startswith("session_id:")
    )
    waited = registry.get("process_manage").call(
        action="wait", session_id=session_id, timeout=3
    )
    assert "exit_code: 0" in waited
    assert "a" in waited and "b" in waited


def test_web_search_unconfigured_and_private_url_rejected(monkeypatch, tmp_path):
    for name in (
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "SORA_WEB_SEARCH_URL",
        "SORA_WEB_SEARCH_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    registry = _registry(tmp_path)
    assert "not configured" in registry.get("web_search").call(query="x")
    assert "refusing" in registry.get("web_extract").call(url="http://127.0.0.1:8000/")


def test_todo_list_and_clarify_gate(tmp_path):
    class Gate:
        def request(self, questions):
            return {
                "outcome": "submitted",
                "responses": [
                    {
                        "question": item["question"],
                        "status": "answered",
                        "user_response": "yes",
                    }
                    for item in questions
                ],
            }

    registry = ToolRegistry()
    register_core_tools(
        registry,
        ToolPackConfig(root=tmp_path),
        clarify_gate=Gate(),
    )
    todo = registry.get("todo_list")
    assert "added" in todo.call(action="add", items=["one", "two"])
    assert "completed" in todo.call(action="complete", ids=[1])
    assert "[x] 1 one" in todo.call(action="list")

    result = registry.get("clarify").call(
        questions=[{"question": "Proceed?", "choices": ["yes", "no"]}]
    )
    assert "submitted" in result
    assert "yes" in result


def test_execute_code(tmp_path):
    registry = _registry(tmp_path)
    result = registry.get("execute_code").call(code="print(21 * 2)")
    assert "exit_code: 0" in result
    assert "42" in result
    assert registry.get("execute_code").needs_approval is True


def test_delegate_task_is_wired_to_an_agent(tmp_path):
    registry = _registry(tmp_path)
    provider = RecordingProvider(["child result"])
    Agent(
        provider=provider,
        registry=registry,
        memory=False,
        identity=False,
        clock=False,
        calendar=False,
    )

    result = registry.get("delegate_task").call(goal="inspect the thing")

    assert "child result" in result
    assert provider.calls


def test_tool_search_finds_new_tools(tmp_path):
    registry = _registry(tmp_path)
    gate = ToolGate(registry, always_visible={"find_tools"})
    finder = FindToolsTool(gate)

    result = finder.call(need="read a file from disk")

    assert "read_file" in result


def test_all_schemas_generate(tmp_path):
    for tool in _registry(tmp_path).all():
        schema = tool.schema
        assert schema["name"] == tool.name
        assert isinstance(schema["parameters"], dict)


def test_approval_flags_preserve_consequential_gates(tmp_path):
    registry = _registry(tmp_path)
    assert registry.get("write_file").needs_approval is True
    assert registry.get("patch").needs_approval is True
    assert registry.get("terminal").needs_approval is True
    assert registry.get("process_manage").needs_approval is True
    assert registry.get("execute_code").needs_approval is True
    assert registry.get("read_file").needs_approval is False
    assert registry.get("search_files").needs_approval is False
    assert registry.get("web_search").needs_approval is False
    assert registry.get("web_extract").needs_approval is False
    assert registry.get("todo_list").needs_approval is False
    assert registry.get("clarify").needs_approval is False
    assert registry.get("delegate_task").needs_approval is False


@pytest.mark.parametrize(
    ("need", "expected"),
    [
        ("read a file", "read_file"),
        ("write a new file", "write_file"),
        ("edit a file", "patch"),
        ("search files for a pattern", "search_files"),
        ("run a shell command", "terminal"),
        ("check a background process", "process_manage"),
        ("search the web", "web_search"),
        ("extract a web page", "web_extract"),
        ("create a task list", "todo_list"),
        ("ask the user a question", "clarify"),
        ("execute python code", "execute_code"),
        ("spawn a subagent", "delegate_task"),
    ],
)
def test_tool_search_ranks_each_core_tool(tmp_path, need, expected):
    registry = _registry(tmp_path)
    gate = ToolGate(registry, always_visible={"find_tools"})
    result = FindToolsTool(gate).call(need=need)
    assert expected in result


def test_chat_handler_wires_delegate_task(tmp_path):
    from sora_agent.channels.chat import ChatHandler

    registry = _registry(tmp_path)
    provider = RecordingProvider(["chat child result"])
    ChatHandler(
        provider,
        registry,
        memory=None,
        identity=False,
        clock=False,
        calendar=False,
    )

    result = registry.get("delegate_task").call(goal="inspect")

    assert "chat child result" in result


def test_static_system_prefix_stays_before_conversation(tmp_path):
    from datetime import datetime, timezone

    from sora_agent.clock import ClockProvider
    from sora_agent.identity import IdentityConfig, IdentityStore
    from sora_agent.state import StateConfig, StateManager
    from sora_agent.state_provider import StatefulProvider

    identity = IdentityStore(IdentityConfig(root=tmp_path / "identity"))
    identity.apply("user", "- Name: Test User")
    state = StateManager("s", StateConfig(root=tmp_path / "state"))
    inner = RecordingProvider(["ok"])
    provider = ClockProvider(
        StatefulProvider(inner, state),
        datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc),
    )
    Agent(
        provider=provider,
        registry=_registry(tmp_path),
        memory=False,
        identity=identity,
        clock=False,
        calendar=False,
    ).run("do work")

    sent = inner.calls[0]
    assert [message.role for message in sent] == [
        "system",
        "system",
        "user",
        "system",
        "system",
    ]
    assert sent[0].content.startswith("You are a careful enterprise-workflow")
    assert sent[1].content.startswith("## PERSISTENT USER PROFILE")
    assert sent[2].content == "do work"
    assert sent[3].content.startswith("## SESSION STATE PROTOCOL")
    assert sent[4].content.startswith("## CURRENT TIME")
