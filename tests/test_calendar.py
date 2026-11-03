"""Tests for the native calendar store, tools, and lazy discovery."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from sora_agent.agent import Agent
from sora_agent.calendar import (
    BALANCED,
    IMPORTANCE_LEVELS,
    IMPORTANT,
    NOT_IMPORTANT,
    CalendarStore,
    default_calendar_db,
    importance_rank,
    normalize_importance,
    parse_moment,
    register_calendar_tools,
)
from sora_agent.channels.chat import ChatHandler
from sora_agent.channels.types import InboundMessage
from sora_agent.providers import Message, ModelTurn, ToolCall
from sora_agent.tool_search import FIND_TOOLS_NAME, FindToolsTool, ToolGate
from sora_agent.tools import ToolExecutor, ToolRegistry

NOW = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)


class TurnProvider:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []
        self.tools: list[list[str]] = []

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls.append(list(messages))
        self.tools.append([tool.name for tool in tools])
        return self.turns.pop(0)


def store_at(path: Path) -> CalendarStore:
    return CalendarStore(path, clock=lambda: NOW)


def _inbound(text: str) -> InboundMessage:
    return InboundMessage(
        channel="fake", chat_id="42", sender_id="7", content=text
    )


def test_default_calendar_db_honors_the_environment(monkeypatch, tmp_path) -> None:
    target = tmp_path / "custom.db"
    monkeypatch.setenv("SORA_CALENDAR_DB", str(target))
    assert default_calendar_db() == target

    monkeypatch.delenv("SORA_CALENDAR_DB")
    assert default_calendar_db() == Path.home() / ".sora_agent" / "calendar.db"


def test_importance_has_exactly_three_levels() -> None:
    assert IMPORTANCE_LEVELS == (IMPORTANT, BALANCED, NOT_IMPORTANT)
    assert normalize_importance(" IMPORTANT ") == IMPORTANT
    assert normalize_importance("balanced") == BALANCED
    assert normalize_importance("not-important") == NOT_IMPORTANT
    assert importance_rank(NOT_IMPORTANT) < importance_rank(BALANCED)
    assert importance_rank(BALANCED) < importance_rank(IMPORTANT)

    with pytest.raises(ValueError, match="importance must be one of"):
        normalize_importance("urgent")


def test_parse_moment_supports_iso_relative_days_and_clock_timezone() -> None:
    assert parse_moment("2026-11-05 14:30", NOW).isoformat() == (
        "2026-11-05T14:30:00+00:00"
    )
    assert parse_moment("2026-11-05T14:30:00Z", NOW).isoformat() == (
        "2026-11-05T14:30:00+00:00"
    )
    assert parse_moment("tomorrow", NOW).isoformat() == (
        "2026-11-04T00:00:00+00:00"
    )
    with pytest.raises(ValueError, match="invalid date/time"):
        parse_moment("not a date", NOW)


def test_store_adds_lists_and_persists_events_and_todos(tmp_path: Path) -> None:
    path = tmp_path / "calendar.db"
    store = store_at(path)
    event = store.add_event(
        "Dentist",
        "2026-11-05T09:00",
        ends_at="2026-11-05T10:00",
        notes="Bring insurance card",
        importance=IMPORTANT,
    )
    todo = store.add_todo(
        "File report",
        due_at="2026-11-04",
        importance=NOT_IMPORTANT,
    )

    assert event.kind == "event"
    assert event.importance == IMPORTANT
    assert event.importance_rank == 3
    assert todo.kind == "todo"
    assert todo.importance == NOT_IMPORTANT
    assert todo.importance_rank == 1

    reopened = store_at(path)
    items = reopened.list_items(start="today", end="2026-12-01")
    assert [item.id for item in items] == [event.id, todo.id]
    assert items[0].title == "Dentist"
    assert items[1].title == "File report"


def test_store_validates_titles_and_event_windows(tmp_path: Path) -> None:
    store = store_at(tmp_path / "calendar.db")

    with pytest.raises(ValueError, match="title is required"):
        store.add_event("   ", "2026-11-05")
    with pytest.raises(ValueError, match="end must not be before"):
        store.add_event(
            "Backwards",
            "2026-11-05T10:00",
            ends_at="2026-11-05T09:00",
        )


def test_store_lists_defaults_filters_and_sorting(tmp_path: Path) -> None:
    store = store_at(tmp_path / "calendar.db")
    low = store.add_event(
        "Later low", "2026-11-06T16:00", importance=NOT_IMPORTANT
    )
    high = store.add_event(
        "Earlier important", "2026-11-06T09:00", importance=IMPORTANT
    )
    outside = store.add_event("Next month", "2026-12-20T09:00")
    due = store.add_todo("Due soon", due_at="2026-11-10")
    undated = store.add_todo("No deadline")
    completed = store.add_todo("Finished", due_at="2026-11-04")
    store.complete_todo(completed.id)

    items = store.list_items()
    ids = [item.id for item in items]
    assert ids == [high.id, low.id, due.id, undated.id]
    assert outside.id not in ids
    assert completed.id not in ids

    assert [item.id for item in store.list_items(kind="events")] == [
        high.id,
        low.id,
    ]
    assert [item.id for item in store.list_items(kind="todos")] == [
        due.id,
        undated.id,
    ]
    assert [
        item.id
        for item in store.list_items(importance=IMPORTANT)
    ] == [high.id]
    assert [
        item.id
        for item in store.list_items(include_completed=True)
    ] == [high.id, low.id, completed.id, due.id, undated.id]


def test_complete_todo_and_set_importance(tmp_path: Path) -> None:
    store = store_at(tmp_path / "calendar.db")
    event = store.add_event("Meeting", "2026-11-05T09:00")
    todo = store.add_todo("Send invoice")

    changed = store.set_importance(todo.id, IMPORTANT)
    assert changed.importance == IMPORTANT
    assert changed.importance_rank == 3

    done = store.complete_todo(todo.id)
    assert done.done is True
    assert done.completed_at
    assert store.complete_todo(todo.id).done is True

    with pytest.raises(ValueError, match="is an event, not a to-do"):
        store.complete_todo(event.id)
    with pytest.raises(ValueError, match="was not found"):
        store.complete_todo(999)
    with pytest.raises(ValueError, match="importance must be one of"):
        store.set_importance(todo.id, "urgent")


def test_calendar_tool_schemas_expose_the_three_stop_slider(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    register_calendar_tools(registry, store_at(tmp_path / "calendar.db"))

    for name in ("calendar_add_event", "calendar_add_todo"):
        schema = registry.get(name).schema
        importance = schema["parameters"]["properties"]["importance"]
        assert importance["enum"] == [
            "important",
            "balanced",
            "not important",
        ]
        assert importance["default"] == "balanced"

    change = registry.get("calendar_set_importance").schema
    assert change["parameters"]["properties"]["importance"]["enum"] == [
        "important",
        "balanced",
        "not important",
    ]


def test_calendar_tools_are_hidden_until_find_tools_loads_them(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path / "calendar.db")
    registry = ToolRegistry()
    register_calendar_tools(registry, store)
    gate = ToolGate(registry, always_visible={FIND_TOOLS_NAME})
    find = FindToolsTool(gate)
    registry.add(find)

    assert gate.visible_names() == [FIND_TOOLS_NAME]
    assert {
        "calendar_list",
        "calendar_add_event",
        "calendar_add_todo",
    } <= {tool.name for tool in gate.hidden()}

    result = find.call(need="calendar events and to-dos")
    assert "calendar_list" in gate.visible_names()
    assert "calendar_add_event" in gate.visible_names()
    assert "calendar_add_todo" in gate.visible_names()
    assert "Loaded tools appear next step" in result

    executor = ToolExecutor(registry)
    added = executor.execute(
        ToolCall(
            id="add_1",
            name="calendar_add_event",
            arguments={
                "title": "Planning",
                "starts_at": "2026-11-05T09:00",
                "importance": "important",
            },
        )
    )
    assert added.executed is True
    assert "Added event" in added.content
    assert store.list_items(kind="events")[0].title == "Planning"


def test_agent_registers_calendar_tools_hidden_by_default(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path / "calendar.db")
    provider = TurnProvider(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="find_1",
                        name=FIND_TOOLS_NAME,
                        arguments={"need": "calendar events"},
                    )
                ]
            ),
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="add_1",
                        name="calendar_add_event",
                        arguments={
                            "title": "Review",
                            "starts_at": "2026-11-05T09:00",
                            "importance": "balanced",
                        },
                    )
                ]
            ),
            ModelTurn(text="Scheduled."),
        ]
    )
    agent = Agent(
        provider=provider,
        memory=False,
        identity=False,
        clock=False,
        calendar=store,
    )

    result = agent.run("Schedule a review.")

    assert agent.registry.has("calendar_list")
    assert provider.tools[0] == [FIND_TOOLS_NAME]
    assert "calendar_add_event" in provider.tools[1]
    assert result.final_answer == "Scheduled."
    assert store.list_items(kind="events")[0].title == "Review"


def test_calendar_can_be_disabled(tmp_path: Path) -> None:
    provider = TurnProvider([ModelTurn(text="Done.")])
    agent = Agent(
        provider=provider,
        memory=False,
        identity=False,
        clock=False,
        calendar=False,
        calendar_db=tmp_path / "unused.db",
    )

    agent.run("hello")

    assert agent.calendar is None
    assert not agent.registry.has("calendar_list")


def test_chat_handler_registers_hidden_calendar_tools(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path / "calendar.db")
    provider = TurnProvider([ModelTurn(text="Hello.")])
    handler = ChatHandler(
        provider,
        max_steps=1,
        clock=False,
        calendar=store,
    )

    assert handler.handle(_inbound("hi")) == "Hello."

    assert handler.registry.has("calendar_list")
    assert provider.tools[0] == [FIND_TOOLS_NAME]

