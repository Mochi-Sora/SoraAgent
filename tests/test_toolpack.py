"""Tests for the built-in Hermes-style core tool pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sora_agent.agent import Agent
from sora_agent.providers import Message, ModelTurn
from sora_agent.runtime_setup import build_registry, resolve_web_search
from sora_agent.settings import advanced_settings, recommended_settings
from sora_agent.tool_search import FindToolsTool, ToolGate
from sora_agent.toolpack import CORE_TOOL_NAMES, ToolPackConfig, register_core_tools
from sora_agent.toolpack.config import WebSearchConfig
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


def _registry(tmp_path: Path, **config) -> ToolRegistry:
    registry = ToolRegistry()
    register_core_tools(registry, ToolPackConfig(root=tmp_path, **config))
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


def test_web_search_disabled_and_private_url_rejected(monkeypatch, tmp_path):
    for name in (
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "SORA_WEB_SEARCH_URL",
        "SORA_WEB_SEARCH_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    registry = _registry(
        tmp_path, web_search=WebSearchConfig(provider="off")
    )
    assert "disabled" in registry.get("web_search").call(query="x")
    assert "refusing" in registry.get("web_extract").call(url="http://127.0.0.1:8000/")


class _FakeHTTPResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status
        self.headers = type(
            "Headers",
            (),
            {"get_content_charset": lambda self: "utf-8"},
        )()

    def read(self, _limit: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_web_search_duckduckgo_decodes_results(monkeypatch, tmp_path):
    html = """
    <html><body>
      <div class="result">
        <a class="result__a"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Frules&amp;rut=x">
          Badminton rules
        </a>
        <a class="result__snippet">The official rules of badminton.</a>
      </div>
      <div class="result">
        <a class="result__a" href="https://example.org/second">Second result</a>
        <div class="result__snippet">Another snippet.</div>
      </div>
    </body></html>
    """
    monkeypatch.setattr(
        "sora_agent.toolpack.web.urlopen",
        lambda request, timeout: _FakeHTTPResponse(html),
    )
    registry = _registry(
        tmp_path, web_search=WebSearchConfig(provider="duckduckgo")
    )

    output = registry.get("web_search").call(query="badminton rules", limit=3)

    assert "Badminton rules" in output
    assert "https://example.com/rules" in output
    assert "The official rules of badminton." in output
    assert "Second result" in output
    assert "https://example.org/second" in output


def test_web_search_duckduckgo_challenge_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sora_agent.toolpack.web.urlopen",
        lambda request, timeout: _FakeHTTPResponse("anomaly challenge", status=202),
    )
    registry = _registry(
        tmp_path, web_search=WebSearchConfig(provider="duckduckgo")
    )

    output = registry.get("web_search").call(query="x")

    assert "DuckDuckGo search failed" in output
    assert "HTTP 202" in output


def test_explicit_provider_without_key_is_reported(tmp_path):
    registry = _registry(
        tmp_path, web_search=WebSearchConfig(provider="tavily")
    )
    assert "TAVILY_API_KEY" in registry.get("web_search").call(query="x")


def test_extra_roots_extend_the_file_sandbox(tmp_path):
    workspace = tmp_path / "workspace"
    extra = tmp_path / "extra"
    workspace.mkdir()
    extra.mkdir()
    registry = _registry(workspace, roots=(extra,))

    assert "wrote" in registry.get("write_file").call(
        path=str(extra / "note.txt"), content="hello"
    )
    assert "hello" in registry.get("read_file").call(path=str(extra / "note.txt"))

    with pytest.raises(ValueError, match="SORA_TOOLPACK_ROOTS"):
        registry.get("write_file").call(
            path=str(tmp_path / "outside" / "note.txt"), content="no"
        )


def test_allow_tmp_opts_the_temp_directory_in(tmp_path):
    import tempfile

    config = ToolPackConfig(root=tmp_path, allow_tmp=True)
    target = Path(tempfile.gettempdir()) / "sora-toolpack-test.txt"
    assert config.resolve(str(target)) == target.resolve()


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
        "system",
        "user",
        "system",
    ]
    assert sent[0].content.startswith("You are a careful enterprise-workflow")
    assert sent[1].content.startswith("## PERSISTENT USER PROFILE")
    assert sent[2].content.startswith("## SESSION STATE PROTOCOL")
    assert sent[3].content == "do work"
    assert sent[4].content.startswith("## CURRENT TIME")


def test_resolve_web_search_prefers_saved_keys_then_auto(monkeypatch):
    for name in (
        "SORA_WEB_SEARCH_PROVIDER",
        "SORA_WEB_SEARCH_URL",
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "SORA_WEB_SEARCH_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = recommended_settings()
    settings.features.web_search.provider = "auto"
    from_secrets = resolve_web_search(settings, {"TAVILY_API_KEY": "tvly-test"})
    assert from_secrets.resolved_provider() == "tavily"
    assert from_secrets.tavily_api_key == "tvly-test"

    settings.features.web_search.provider = "duckduckgo"
    explicit = resolve_web_search(settings, {"TAVILY_API_KEY": "tvly-test"})
    assert explicit.resolved_provider() == "duckduckgo"


def test_registry_passes_saved_search_key(monkeypatch):
    settings = recommended_settings()
    settings.features.web_search.provider = "tavily"
    captured: dict[str, Any] = {}

    def fake_http_json(url, *, payload=None, headers=None, timeout=20.0):
        captured.update(payload or {})
        return {
            "results": [
                {
                    "title": "Rules",
                    "url": "https://example.com/rules",
                    "content": "The rules.",
                }
            ]
        }

    monkeypatch.setattr("sora_agent.toolpack.web._http_json", fake_http_json)
    registry = build_registry(settings, secrets={"TAVILY_API_KEY": "tvly-saved"})

    output = registry.get("web_search").call(query="rules")

    assert "Rules" in output
    assert captured["api_key"] == "tvly-saved"


def test_root_environment_variables(monkeypatch, tmp_path):
    import tempfile

    workspace = tmp_path / "workspace"
    extra = tmp_path / "extra"
    workspace.mkdir()
    extra.mkdir()
    monkeypatch.setenv("SORA_TOOLPACK_ROOTS", str(extra))
    monkeypatch.setenv("SORA_TOOLPACK_ALLOW_TMP", "1")

    config = ToolPackConfig.from_env(root=workspace)

    assert extra.resolve() in config.roots
    assert Path(tempfile.gettempdir()).resolve() in config.roots


def test_web_redirect_guard_rejects_private_destinations() -> None:
    import urllib.request

    from sora_agent.toolpack.web import _PublicRedirectHandler

    handler = _PublicRedirectHandler(allow_private=False)
    request = urllib.request.Request("https://example.com/start")
    with pytest.raises(ValueError, match="private, loopback"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1:8080/secret",
        )


def test_search_files_does_not_follow_symlinks_outside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("TOP_SECRET_VALUE\n", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    registry = _registry(workspace)
    output = registry.get("search_files").call(
        pattern="TOP_SECRET_VALUE", path=".", target="content"
    )
    assert "link.txt:" not in output
    assert "no matches" in output.lower()


def test_search_files_regex_timeout_is_enforced_in_a_child_process(tmp_path):
    from sora_agent.isolation import ToolTimeout

    target = tmp_path / "payload.txt"
    target.write_text("a" * 40 + "!\n", encoding="utf-8")
    registry = _registry(tmp_path, search_timeout=0.3)
    tool = registry.get("search_files")
    with pytest.raises(ToolTimeout):
        tool.call(pattern="(a+)+$", path=".", target="content")


def test_public_peer_guard_rejects_loopback_socket() -> None:
    from sora_agent.toolpack.safety import assert_public_peer

    class Sock:
        def getpeername(self):
            return ("127.0.0.1", 1234)

    class Raw:
        _sock = Sock()

    class Fp:
        raw = Raw()

    class Response:
        fp = Fp()

    with pytest.raises(ValueError, match="private, loopback"):
        assert_public_peer(Response(), "https://public.example")


def test_shell_output_is_bounded_while_running(tmp_path):
    registry = _registry(tmp_path, max_terminal_output=200)
    output = registry.get("terminal").call(
        command='python3 -c "print(\'x\' * 5000)"'
    )
    assert "exit_code: 0" in output
    assert len(output) < 500

    started = registry.get("terminal").call(
        command='python3 -c "print(\'y\' * 5000)"',
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
    assert len(waited) < 600


def test_web_extract_runs_in_a_child_process(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><title>Test</title><body><p>Hello web</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = _registry(tmp_path, allow_private_network=True)
        output = registry.get("web_extract").call(
            url=f"http://127.0.0.1:{server.server_address[1]}/"
        )
        assert "Hello web" in output
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_execute_code_output_is_bounded(tmp_path):
    registry = _registry(tmp_path, max_terminal_output=200)
    output = registry.get("execute_code").call(
        code="print('z' * 5000)"
    )
    assert "exit_code: 0" in output
    assert len(output) < 500


def test_shell_process_is_killed_once_output_cap_is_reached(tmp_path):
    import time

    registry = _registry(tmp_path, max_terminal_output=200, terminal_timeout=30)
    started = time.monotonic()
    output = registry.get("terminal").call(
        command="yes q",
        timeout=30,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 10
    assert "truncated" in output
