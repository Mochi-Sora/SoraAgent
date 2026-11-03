"""A small, native calendar the model can browse through tools.

The calendar is deliberately local and dependency-free: one SQLite file at
``~/.sora_agent/calendar.db`` (override with ``SORA_CALENDAR_DB``).  It stores
two kinds of items:

* ``event`` — a title with a start and optional end
* ``todo``  — a task with an optional due date and a completed flag

Both carry an importance level with exactly three stops: ``important``,
``balanced``, and ``not important``.  The level is stored alongside a numeric
rank so lists can be sorted predictably.

Tools are registered like any other tool.  They stay hidden until the model
discovers them through ``find_tools``, which keeps the initial prompt small.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from .clock import default_clock, resolve_clock
from .tools import Tool, ToolRegistry

IMPORTANT = "important"
BALANCED = "balanced"
NOT_IMPORTANT = "not important"
IMPORTANCE_LEVELS = (IMPORTANT, BALANCED, NOT_IMPORTANT)
IMPORTANCE_RANKS = {
    NOT_IMPORTANT: 1,
    BALANCED: 2,
    IMPORTANT: 3,
}
EVENT = "event"
TODO = "todo"
ITEM_KINDS = (EVENT, TODO)
DEFAULT_CALENDAR_DB = "~/.sora_agent/calendar.db"

_IMPORTANCE_ALIASES = {
    "important": IMPORTANT,
    "high": IMPORTANT,
    "critical": IMPORTANT,
    "balanced": BALANCED,
    "normal": BALANCED,
    "medium": BALANCED,
    "not important": NOT_IMPORTANT,
    "notimportant": NOT_IMPORTANT,
    "unimportant": NOT_IMPORTANT,
    "low": NOT_IMPORTANT,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    starts_at TEXT,
    ends_at TEXT,
    due_at TEXT,
    notes TEXT NOT NULL DEFAULT '',
    importance TEXT NOT NULL,
    importance_rank INTEGER NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_calendar_kind ON calendar_items(kind);
CREATE INDEX IF NOT EXISTS idx_calendar_start ON calendar_items(starts_at);
CREATE INDEX IF NOT EXISTS idx_calendar_due ON calendar_items(due_at);
"""


def default_calendar_db() -> Path:
    """Return the configured calendar database path."""
    configured = os.getenv("SORA_CALENDAR_DB", "").strip()
    return Path(configured or DEFAULT_CALENDAR_DB).expanduser()


def normalize_importance(value: Any) -> str:
    """Return one of the three canonical importance levels."""
    text = " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )
    level = _IMPORTANCE_ALIASES.get(text)
    if level is None:
        raise ValueError(
            "importance must be one of: "
            + ", ".join(repr(item) for item in IMPORTANCE_LEVELS)
        )
    return level


def importance_rank(value: Any) -> int:
    """Return the sort rank for an importance level (higher is more important)."""
    return IMPORTANCE_RANKS[normalize_importance(value)]


def _now(clock: Any = None) -> datetime:
    reader = resolve_clock(clock) or default_clock
    moment = reader()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment


def parse_moment(value: Any, clock: Any = None) -> datetime:
    """Parse a date/time into an aware ``datetime``.

    ISO 8601 is accepted (``2026-11-05``, ``2026-11-05 14:30``,
    ``2026-11-05T14:30:00+08:00``) along with ``now``, ``today``, ``tomorrow``,
    and ``yesterday``.  Naive values are interpreted in the clock's timezone.
    """
    now = _now(clock)
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("a date/time is required")
        lowered = " ".join(text.lower().split())
        if lowered == "now":
            moment = now
        elif lowered in {"today", "tomorrow", "yesterday"}:
            days = {"today": 0, "tomorrow": 1, "yesterday": -1}[lowered]
            base = now + timedelta(days=days)
            moment = base.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            normalized = text[:-1] + "+00:00" if text.lower().endswith("z") else text
            try:
                moment = datetime.fromisoformat(normalized)
            except ValueError:
                try:
                    moment = datetime.fromisoformat(
                        normalized.replace(" ", "T", 1)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"invalid date/time {value!r}; use ISO 8601, for example "
                        "2026-11-05 or 2026-11-05T14:30"
                    ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=now.tzinfo)
    return moment


def _display(value: str) -> str:
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return value
    return moment.strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True)
class CalendarItem:
    """One event or to-do."""

    id: int
    kind: str
    title: str
    starts_at: str = ""
    ends_at: str = ""
    due_at: str = ""
    notes: str = ""
    importance: str = BALANCED
    importance_rank: int = IMPORTANCE_RANKS[BALANCED]
    done: bool = False
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""

    @property
    def is_event(self) -> bool:
        return self.kind == EVENT

    @property
    def is_todo(self) -> bool:
        return self.kind == TODO


class CalendarStore:
    """SQLite-backed events and to-dos."""

    def __init__(
        self, path: str | Path | None = None, clock: Any = None
    ) -> None:
        self.path = Path(path).expanduser() if path else default_calendar_db()
        self.clock = resolve_clock(clock) or default_clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _stamp(self) -> str:
        return _now(self.clock).astimezone(timezone.utc).isoformat(
            timespec="seconds"
        )

    def get(self, item_id: int) -> Optional[CalendarItem]:  # noqa: UP045
        """Return one item by id, or ``None``."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM calendar_items WHERE id = ?", (int(item_id),)
            ).fetchone()
        return _row_to_item(row) if row is not None else None

    def add_event(
        self,
        title: str,
        starts_at: Any,
        ends_at: Any = None,
        notes: str = "",
        importance: Any = BALANCED,
    ) -> CalendarItem:
        start = parse_moment(starts_at, self.clock)
        end = parse_moment(ends_at, self.clock) if ends_at else None
        if end is not None and end < start:
            raise ValueError("event end must not be before its start")
        return self._insert(
            EVENT,
            title,
            starts_at=start.isoformat(timespec="seconds"),
            ends_at=end.isoformat(timespec="seconds") if end else "",
            notes=notes,
            importance=importance,
        )

    def add_todo(
        self,
        title: str,
        due_at: Any = None,
        notes: str = "",
        importance: Any = BALANCED,
    ) -> CalendarItem:
        due = parse_moment(due_at, self.clock) if due_at else None
        return self._insert(
            TODO,
            title,
            due_at=due.isoformat(timespec="seconds") if due else "",
            notes=notes,
            importance=importance,
        )

    def _insert(
        self,
        kind: str,
        title: str,
        *,
        starts_at: str = "",
        ends_at: str = "",
        due_at: str = "",
        notes: str = "",
        importance: Any = BALANCED,
    ) -> CalendarItem:
        clean_title = " ".join(str(title or "").split())
        if not clean_title:
            raise ValueError("a title is required")
        level = normalize_importance(importance)
        stamp = self._stamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO calendar_items (
                    kind, title, starts_at, ends_at, due_at, notes,
                    importance, importance_rank, done, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    kind,
                    clean_title,
                    starts_at or None,
                    ends_at or None,
                    due_at or None,
                    str(notes or "").strip(),
                    level,
                    IMPORTANCE_RANKS[level],
                    stamp,
                    stamp,
                ),
            )
            item_id = int(cursor.lastrowid or 0)
        item = self.get(item_id)
        if item is None:  # pragma: no cover - the insert just committed
            raise RuntimeError("calendar item was not stored")
        return item

    def complete_todo(self, item_id: int) -> CalendarItem:
        item = self.get(item_id)
        if item is None:
            raise ValueError(f"calendar item #{item_id} was not found")
        if item.kind != TODO:
            raise ValueError(
                f"calendar item #{item_id} is an event, not a to-do"
            )
        if item.done:
            return item
        stamp = self._stamp()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE calendar_items
                SET done = 1, completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (stamp, stamp, int(item_id)),
            )
        updated = self.get(item_id)
        if updated is None:  # pragma: no cover - the update just committed
            raise RuntimeError("calendar item disappeared")
        return updated

    def set_importance(self, item_id: int, importance: Any) -> CalendarItem:
        level = normalize_importance(importance)
        if self.get(item_id) is None:
            raise ValueError(f"calendar item #{item_id} was not found")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE calendar_items
                SET importance = ?, importance_rank = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    level,
                    IMPORTANCE_RANKS[level],
                    self._stamp(),
                    int(item_id),
                ),
            )
        updated = self.get(item_id)
        if updated is None:  # pragma: no cover - the update just committed
            raise RuntimeError("calendar item disappeared")
        return updated

    def list_items(
        self,
        start: Any = None,
        end: Any = None,
        kind: str = "all",
        include_completed: bool = False,
        importance: Any = None,
        limit: int = 200,
    ) -> list[CalendarItem]:
        """Return items in a time window, grouped events first, then to-dos."""
        kind_key = _normalize_kind(kind)
        level = normalize_importance(importance) if importance else None
        explicit_window = bool(start) or bool(end)

        now = _now(self.clock)
        if start:
            window_start = parse_moment(start, self.clock)
        elif end:
            window_start = parse_moment(end, self.clock) - timedelta(days=30)
        else:
            window_start = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        window_end = (
            parse_moment(end, self.clock)
            if end
            else window_start + timedelta(days=30)
        )
        if window_end < window_start:
            raise ValueError("calendar window end must not be before its start")

        query = "SELECT * FROM calendar_items WHERE 1 = 1"
        params: list[Any] = []
        if kind_key != "all":
            query += " AND kind = ?"
            params.append(kind_key)
        if not include_completed:
            query += " AND done = 0"
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        items = [_row_to_item(row) for row in rows]

        selected: list[CalendarItem] = []
        for item in items:
            if level is not None and item.importance != level:
                continue
            if item.kind == EVENT:
                item_start = parse_moment(item.starts_at, self.clock)
                item_end = (
                    parse_moment(item.ends_at, self.clock)
                    if item.ends_at
                    else item_start
                )
                if item_end < window_start or item_start > window_end:
                    continue
            elif item.due_at and explicit_window:
                due = parse_moment(item.due_at, self.clock)
                if due < window_start or due > window_end:
                    continue
            selected.append(item)

        events = [item for item in selected if item.kind == EVENT]
        todos = [item for item in selected if item.kind == TODO]
        events.sort(
            key=lambda item: (
                parse_moment(item.starts_at, self.clock),
                -item.importance_rank,
            )
        )
        todos.sort(
            key=lambda item: (
                parse_moment(item.due_at, self.clock)
                if item.due_at
                else datetime.max.replace(tzinfo=timezone.utc),
                -item.importance_rank,
            )
        )
        combined = events + todos
        if limit > 0:
            return combined[:limit]
        return combined


def _normalize_kind(value: Any) -> str:
    text = " ".join(
        str(value or "all")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )
    if text in {"all", ""}:
        return "all"
    if text in {"event", "events"}:
        return EVENT
    if text in {"todo", "todos", "to do", "to dos", "task", "tasks"}:
        return TODO
    raise ValueError("kind must be 'all', 'events', or 'todos'")


def _row_to_item(row: sqlite3.Row) -> CalendarItem:
    return CalendarItem(
        id=int(row["id"]),
        kind=str(row["kind"]),
        title=str(row["title"]),
        starts_at=str(row["starts_at"] or ""),
        ends_at=str(row["ends_at"] or ""),
        due_at=str(row["due_at"] or ""),
        notes=str(row["notes"] or ""),
        importance=str(row["importance"]),
        importance_rank=int(row["importance_rank"]),
        done=bool(row["done"]),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
        completed_at=str(row["completed_at"] or ""),
    )


def render_item(item: CalendarItem) -> str:
    """Render one item as a single line the model can read back."""
    if item.is_event:
        when = _display(item.starts_at)
        if item.ends_at:
            when += f" to {_display(item.ends_at)}"
    else:
        when = f"due {_display(item.due_at)}" if item.due_at else "no due date"
        if item.done:
            when += " (done)"
    parts = [
        f"#{item.id}",
        item.kind,
        when,
        f"importance={item.importance}",
        item.title,
    ]
    if item.notes:
        parts.append(f"notes: {item.notes}")
    return "- " + " | ".join(parts)


def _render_section(label: str, items: Sequence[CalendarItem]) -> list[str]:
    if not items:
        return []
    lines = [f"{label} ({len(items)}):"]
    lines.extend(render_item(item) for item in items)
    return lines


def resolve_calendar(
    calendar: Any = None, db_path: Any = None
) -> Any:
    """Resolve the ``calendar`` option into a store, or ``None`` if disabled."""
    if calendar is False:
        return None
    if isinstance(calendar, (str, Path)):
        return CalendarStore(calendar)
    if calendar is None or calendar is True or calendar == "default":
        return CalendarStore(db_path)
    return calendar


def default_calendar(db_path: Any = None) -> CalendarStore:
    """Return the default calendar store."""
    return CalendarStore(db_path)


def register_calendar_tools(
    registry: ToolRegistry, store: Any
) -> list[Tool]:
    """Register the calendar tool set on ``registry``.

    Tools are ordinary hidden tools: ``AgentRuntime`` keeps them out of the
    initial tool list, so the model reaches them through ``find_tools``.
    Existing tools with the same names are left untouched.
    """

    def calendar_list(
        start: Annotated[
            Optional[str],  # noqa: UP045
            "Window start as ISO 8601 (2026-11-05) or 'today'/'tomorrow'. "
            "Omit for today onward.",
        ] = None,
        end: Annotated[
            Optional[str],  # noqa: UP045
            "Window end as ISO 8601 or a relative day. Omit for 30 days.",
        ] = None,
        kind: Annotated[
            Literal["all", "events", "todos"],
            "Which items to list.",
        ] = "all",
        include_completed: Annotated[
            bool, "Include already completed to-dos."
        ] = False,
        importance: Annotated[
            Optional[  # noqa: UP045
                Literal["important", "balanced", "not important"]
            ],
            "Only list this importance level.",
        ] = None,
        limit: Annotated[
            int, "Maximum items per section (1-200)."
        ] = 50,
    ) -> str:
        kind_key = _normalize_kind(kind)
        capped = max(1, min(int(limit), 200))
        common = {
            "start": start,
            "end": end,
            "include_completed": include_completed,
            "importance": importance,
            "limit": capped,
        }
        lines: list[str] = []
        if kind_key in {"all", EVENT}:
            events = store.list_items(kind=EVENT, **common)
            section = _render_section("Events", events)
            if section:
                lines.extend(section)
            elif kind_key == EVENT:
                lines.append("No events found.")
        if kind_key in {"all", TODO}:
            todos = store.list_items(kind=TODO, **common)
            section = _render_section("To-dos", todos)
            if section:
                lines.extend(section)
            elif kind_key == TODO:
                lines.append("No to-dos found.")
        return "\n".join(lines) if lines else "No calendar items found."

    def calendar_add_event(
        title: Annotated[str, "Short event title."],
        starts_at: Annotated[
            str,
            "Start date/time in ISO 8601 or 'today'/'tomorrow'.",
        ],
        ends_at: Annotated[
            Optional[str],  # noqa: UP045
            "Optional end date/time."
        ] = None,
        importance: Annotated[
            Literal["important", "balanced", "not important"],
            "Three-level importance slider.",
        ] = "balanced",
        notes: Annotated[str, "Optional details."] = "",
    ) -> str:
        item = store.add_event(
            title,
            starts_at,
            ends_at=ends_at,
            notes=notes,
            importance=importance,
        )
        when = _display(item.starts_at)
        if item.ends_at:
            when += f" to {_display(item.ends_at)}"
        return (
            f"Added event #{item.id}: {item.title} ({when}, "
            f"importance={item.importance})."
        )

    def calendar_add_todo(
        title: Annotated[str, "Short task or reminder title."],
        due_at: Annotated[
            Optional[str],  # noqa: UP045
            "Optional due date/time in ISO 8601."
        ] = None,
        importance: Annotated[
            Literal["important", "balanced", "not important"],
            "Three-level importance slider.",
        ] = "balanced",
        notes: Annotated[str, "Optional details."] = "",
    ) -> str:
        item = store.add_todo(
            title,
            due_at=due_at,
            notes=notes,
            importance=importance,
        )
        due = (
            f"due {_display(item.due_at)}"
            if item.due_at
            else "no due date"
        )
        return (
            f"Added to-do #{item.id}: {item.title} ({due}, "
            f"importance={item.importance})."
        )

    def calendar_complete_todo(
        item_id: Annotated[int, "To-do id shown by calendar_list."],
    ) -> str:
        item = store.get(item_id)
        if item is None:
            raise ValueError(f"calendar item #{item_id} was not found")
        if item.kind != TODO:
            raise ValueError(
                f"calendar item #{item_id} is an event, not a to-do"
            )
        if item.done:
            return f"To-do #{item.id} is already completed: {item.title}."
        store.complete_todo(item_id)
        return f"Completed to-do #{item.id}: {item.title}."

    def calendar_set_importance(
        item_id: Annotated[int, "Event or to-do id."],
        importance: Annotated[
            Literal["important", "balanced", "not important"],
            "Three-level importance slider.",
        ],
    ) -> str:
        item = store.set_importance(item_id, importance)
        return (
            f"Set {item.kind} #{item.id} importance to "
            f"{item.importance}: {item.title}."
        )

    definitions = (
        (
            calendar_list,
            "calendar_list",
            (
                "List calendar events, meetings, and to-dos in a date window. "
                "Use this to browse the schedule, check what is planned, or "
                "find an item id. Dates accept ISO 8601 and relative days; "
                "omit them for upcoming events and all open to-dos."
            ),
        ),
        (
            calendar_add_event,
            "calendar_add_event",
            (
                "Add an event or meeting to the calendar with a title, start, "
                "optional end, notes, and an importance level: important, "
                "balanced, or not important."
            ),
        ),
        (
            calendar_add_todo,
            "calendar_add_todo",
            (
                "Add a to-do task or reminder to the calendar with an optional "
                "due date, notes, and an importance level: important, "
                "balanced, or not important."
            ),
        ),
        (
            calendar_complete_todo,
            "calendar_complete_todo",
            (
                "Mark a calendar to-do as done or completed. Get the to-do id "
                "from calendar_list."
            ),
        ),
        (
            calendar_set_importance,
            "calendar_set_importance",
            (
                "Move an event or to-do's importance slider to important, "
                "balanced, or not important."
            ),
        ),
    )
    added: list[Tool] = []
    for function, name, description in definitions:
        if registry.has(name):
            continue
        added.append(registry.add(Tool(function, name, description)))
    return added


def format_calendar(items: Sequence[CalendarItem]) -> str:
    """Render a mixed list of items, for tests and small integrations."""
    events = [item for item in items if item.kind == EVENT]
    todos = [item for item in items if item.kind == TODO]
    lines = _render_section("Events", events) + _render_section(
        "To-dos", todos
    )
    return "\n".join(lines) if lines else "No calendar items found."


__all__ = [
    "BALANCED",
    "DEFAULT_CALENDAR_DB",
    "EVENT",
    "IMPORTANCE_LEVELS",
    "IMPORTANCE_RANKS",
    "IMPORTANT",
    "ITEM_KINDS",
    "NOT_IMPORTANT",
    "TODO",
    "CalendarItem",
    "CalendarStore",
    "default_calendar",
    "default_calendar_db",
    "format_calendar",
    "importance_rank",
    "normalize_importance",
    "parse_moment",
    "register_calendar_tools",
    "render_item",
    "resolve_calendar",
]
