"""Tests for the persistent specialist registry and tools."""

from __future__ import annotations

import pytest

from sora_agent.delegation import (
    DelegationAgent,
    DelegationError,
    DelegationStore,
    build_specialist_registry,
    rank,
    register_specialist_tools,
)
from sora_agent.runtime_setup import desktop_settings
from sora_agent.tools import ToolRegistry


def _store(tmp_path) -> DelegationStore:
    return DelegationStore(tmp_path / "delegations")


def _agent(agent_id="research", **overrides) -> DelegationAgent:
    values = {
        "id": agent_id,
        "name": "Research Analyst",
        "description": "Finds public sources",
        "when_to_use": "research, sources, fact checking",
        "prompt": "Find and cite reliable sources.",
        "tools": ("research",),
    }
    values.update(overrides)
    return DelegationAgent(**values)


def test_store_create_list_and_delete(tmp_path):
    store = _store(tmp_path)
    agent = store.create(
        name="Research Analyst",
        description="Finds public sources",
        when_to_use="research and citations",
        prompt="Find sources.",
        tools=("research",),
    )
    assert agent.id == "research-analyst"
    loaded = store.get("research-analyst")
    assert loaded is not None
    assert loaded.tools == ("research",)
    assert (store.root / agent.id / "workspace").is_dir()
    assert [item.id for item in store.list()] == ["research-analyst"]
    assert store.delete("research-analyst") is True
    assert store.get("research-analyst") is None


def test_store_rejects_forbidden_tools_and_bad_ids(tmp_path):
    store = _store(tmp_path)
    for tool in ("find_delegates", "delegate_to_agent"):
        try:
            store.create(
                name="Bad",
                description="Bad",
                prompt="Bad.",
                tools=(tool,),
            )
        except DelegationError as exc:
            assert "delegation control" in str(exc)
        else:
            raise AssertionError(f"{tool} should be rejected")
    try:
        _agent("../escape").validate()
    except DelegationError:
        pass
    else:
        raise AssertionError("invalid id should be rejected")


def test_router_prefers_the_matching_specialist(tmp_path):
    store = _store(tmp_path)
    store.create(
        name="Research Analyst",
        description="Finds public sources",
        when_to_use="research, citations, literature",
        prompt="Research.",
        tools=("research",),
    )
    store.create(
        name="Python Fixer",
        description="Repairs Python tests",
        when_to_use="python, pytest, traceback, code",
        prompt="Fix code.",
        tools=("code",),
    )
    agents = store.list()
    matches = rank(agents, "find sources and citations", 2)
    assert matches
    assert matches[0][0].id == "research-analyst"


def test_tools_find_and_invoke_with_a_runner(tmp_path):
    store = _store(tmp_path)
    store.create(
        name="Research Analyst",
        description="Finds public sources",
        when_to_use="research, citations",
        prompt="Research.",
        tools=("research",),
    )
    calls = []

    def runner(agent, goal, context="", *, approval=None):
        calls.append((agent.id, goal, context, approval))
        return "a cited answer"

    registry = ToolRegistry()
    register_specialist_tools(registry, store, runner=runner)
    found = registry.get("find_delegates").fn("find sources")
    assert "research-analyst" in found
    assert "delegate_to_agent" in found

    result = registry.get("delegate_to_agent").fn(
        "research-analyst", "Summarize the topic", "Prefer primary sources"
    )
    assert "a cited answer" in result
    assert calls[0][0] == "research-analyst"
    assert "Prefer primary sources" in result or calls[0][2] == "Prefer primary sources"


def test_tools_report_unknown_specialist(tmp_path):
    store = _store(tmp_path)
    registry = ToolRegistry()
    register_specialist_tools(registry, store, runner=lambda *args, **kwargs: "x")
    result = registry.get("delegate_to_agent").fn("missing", "do work")
    assert "unknown specialist" in result


def test_specialist_registry_is_scoped_and_always_has_workers():
    settings = desktop_settings()
    settings.features.tools = True
    agent = _agent(tools=("research",))
    registry = build_specialist_registry(settings, {}, agent)
    names = {tool.name for tool in registry.all()}
    assert "web_search" in names
    assert "delegate_task" in names
    assert "find_delegates" not in names
    assert "delegate_to_agent" not in names
    assert "terminal" not in names


def test_specialist_with_only_worker_tool_is_rejected():
    settings = desktop_settings()
    settings.features.tools = True
    agent = _agent(tools=("delegate_task",))
    try:
        build_specialist_registry(settings, {}, agent)
    except DelegationError as exc:
        assert "no usable tools" in str(exc)
    else:
        raise AssertionError("a worker-only specialist should be rejected")


class _PlainProvider:
    def __init__(self, text: str = "specialist answer") -> None:
        self.text = text
        self.calls = []

    def complete_turn(self, messages, tools):
        from sora_agent.providers import ModelTurn

        self.calls.append((list(messages), list(tools)))
        return ModelTurn(text=self.text)


def test_agent_accepts_a_specialist_system_prompt():
    from sora_agent.agent import Agent

    provider = _PlainProvider("ok")
    agent = Agent(
        provider=provider,
        registry=ToolRegistry(),
        system_prompt="You are a research specialist.",
        delegate=False,
    )
    result = agent.run("do the task")
    assert result.final_answer == "ok"
    assert "You are a research specialist." in provider.calls[0][0][0].content


def test_specialist_runner_builds_a_child_agent(monkeypatch, tmp_path):
    from sora_agent.delegation import SpecialistAgentRunner, factory

    provider = _PlainProvider("a cited answer")
    monkeypatch.setattr(factory, "build_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(
        factory, "build_specialist_registry", lambda *args, **kwargs: ToolRegistry()
    )
    settings = desktop_settings()
    runner = SpecialistAgentRunner(
        settings, {}, store=DelegationStore(tmp_path / "delegations")
    )
    result = runner(
        _agent(),
        "Summarize the evidence",
        "Prefer primary sources",
    )
    assert result == "a cited answer"
    system = provider.calls[0][0][0].content
    assert "Research Analyst" in system
    assert "Find and cite reliable sources." in system


def test_chat_handler_exposes_specialist_routing_tools(tmp_path):
    from sora_agent.channels.chat import ChatHandler
    from sora_agent.channels.sessions import SessionStore
    from sora_agent.channels.types import InboundMessage
    from sora_agent.tools import Tool

    provider = _PlainProvider("hi")
    registry = ToolRegistry()
    registry.add(Tool(lambda: "none", "find_delegates", "Find specialists"))
    registry.add(Tool(lambda: "none", "delegate_to_agent", "Ask a specialist"))
    handler = ChatHandler(
        provider,
        registry,
        sessions=SessionStore(tmp_path / "sessions"),
        delegate=False,
    )
    handler.handle(InboundMessage("cli", "local", "42", "hello"))

    visible = {tool.name for tool in provider.calls[0][1]}
    assert "find_delegates" in visible
    assert "delegate_to_agent" in visible


def test_chat_handler_can_call_a_specialist_tool(tmp_path):
    from sora_agent.channels.chat import ChatHandler
    from sora_agent.channels.sessions import SessionStore
    from sora_agent.channels.types import InboundMessage
    from sora_agent.providers import ModelTurn, ToolCall

    store = _store(tmp_path)
    store.create(
        name="Research Analyst",
        description="Finds public sources",
        when_to_use="research, citations",
        prompt="Research.",
        tools=("research",),
    )
    registry = ToolRegistry()
    register_specialist_tools(
        registry,
        store,
        runner=lambda *args, **kwargs: "specialist evidence",
    )

    class Provider:
        def __init__(self):
            self.calls = 0
            self.seen = []

        def complete_turn(self, messages, tools):
            self.calls += 1
            self.seen.append(list(messages))
            if self.calls == 1:
                return ModelTurn(
                    tool_calls=[
                        ToolCall(
                            "c1",
                            "delegate_to_agent",
                            {
                                "agent": "research-analyst",
                                "goal": "Find evidence",
                            },
                        )
                    ]
                )
            return ModelTurn(text="final answer")

    provider = Provider()
    handler = ChatHandler(
        provider,
        registry,
        sessions=SessionStore(tmp_path / "sessions"),
        delegate=False,
    )
    reply = handler.handle(InboundMessage("cli", "local", "42", "research this"))
    assert "final answer" in reply
    tool_messages = [
        message.content
        for message in provider.seen[1]
        if getattr(message, "role", "") == "tool"
    ]
    assert any("specialist evidence" in text for text in tool_messages)


def test_auto_create_tool_is_opt_in_and_restricted(tmp_path):
    from sora_agent.runtime_setup import build_registry

    settings = desktop_settings()
    settings.features.tools = True
    store = _store(tmp_path)
    registry = build_registry(
        settings,
        delegation_store=store,
        allow_create=True,
    )
    assert registry.has("create_delegation_agent")
    result = registry.get("create_delegation_agent").fn(
        name="Doc Reader",
        description="Reads repository docs",
        when="read documentation, summarize files",
        prompt="Read docs and summarize.",
        tools="research,terminal",
    )
    assert "Created provisional specialist" in result
    agent = store.get("doc-reader")
    assert agent is not None
    assert agent.created_by == "main"
    assert "terminal" not in agent.resolved_tools()
    assert "web_search" in agent.resolved_tools()
    assert agent.metadata.get("provisional") is True


def test_auto_create_is_disabled_by_default(tmp_path):
    from sora_agent.runtime_setup import build_registry

    settings = desktop_settings()
    settings.features.tools = True
    registry = build_registry(settings, delegation_store=_store(tmp_path))
    assert not registry.has("create_delegation_agent")


def test_delegation_settings_round_trip():
    from sora_agent.settings import FeatureSettings, Settings

    settings = Settings.from_dict(
        {"setup": {"mode": "advanced"}, "features": {"delegation": {"create_mode": "auto"}}}
    )
    assert settings.features.delegation.create_mode == "auto"
    assert FeatureSettings.from_dict({"delegation": {"max_per_day": 1}}).delegation.max_per_day == 1


def test_specialist_workspace_scopes_file_tools(tmp_path):
    settings = desktop_settings()
    settings.features.tools = True
    agent = _agent(tools=("files",))
    workspace = tmp_path / "specialist"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("inside data", encoding="utf-8")
    registry = build_specialist_registry(
        settings, {}, agent, workspace=workspace
    )
    assert "inside data" in registry.get("read_file").fn("inside.txt")
    with pytest.raises(ValueError, match="outside the configured workspace"):
        registry.get("read_file").fn("../outside.txt")


def test_specialist_daily_store_persists_in_workspace(tmp_path):
    settings = desktop_settings()
    settings.features.tools = True
    agent = _agent(tools=("research",))
    workspace = tmp_path / "specialist"
    registry = build_specialist_registry(
        settings, {}, agent, workspace=workspace
    )
    assert "daily log" in registry.get("remember_daily").fn(
        "Primary source: example.org"
    )
    assert list((workspace / "daily").glob("*.db"))
    recalled = registry.get("recall_daily").fn("primary source")
    assert "example.org" in recalled
    assert not (workspace / "notes.jsonl").exists()


def test_specialist_can_write_inside_its_workspace(tmp_path):
    settings = desktop_settings()
    settings.features.tools = True
    agent = _agent(tools=("files",))
    workspace = tmp_path / "specialist"
    registry = build_specialist_registry(
        settings, {}, agent, workspace=workspace
    )
    registry.get("write_file").fn("report.md", "# Report\n")
    assert (workspace / "report.md").read_text(encoding="utf-8") == "# Report\n"
    with pytest.raises(ValueError, match="outside the configured workspace"):
        registry.get("write_file").fn("../escape.md", "no")


def test_specialist_daily_notes_are_redacted(tmp_path):
    settings = desktop_settings()
    settings.features.tools = True
    agent = _agent(tools=("research",))
    workspace = tmp_path / "specialist"
    registry = build_specialist_registry(
        settings,
        {},
        agent,
        workspace=workspace,
        redactor=lambda text: text.replace("secret-value", "[secret]"),
    )
    registry.get("remember_daily").fn("token=secret-value")
    recalled = registry.get("recall_daily").fn("token")
    assert "secret-value" not in recalled
    assert "[secret]" in recalled
