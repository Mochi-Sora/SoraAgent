"""Short-lived verbatim conversation memory.

One SQLite database per UTC day, searched with FTS5 when SQLite provides it
and with a small token index otherwise. Only user and assistant messages are
written here; tool output and system prompts are deliberately excluded.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence

_DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.db$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


@dataclass(frozen=True)
class DailyHit:
    day: date
    message_id: int
    ts: str
    role: str
    content: str
    score: float
    source: str = "daily"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def day_string(day: date) -> str:
    return day.isoformat()


class DailyStore:
    def __init__(
        self,
        root: Path,
        session_id: str = "default",
        clock: Optional[Callable[[], datetime]] = None,
        use_fts: bool = True,
    ) -> None:
        self.root = Path(root)
        self.session_id = session_id
        self.clock = clock or utcnow
        self.use_fts = use_fts
        self.root.mkdir(parents=True, exist_ok=True)

    def now(self) -> datetime:
        return _as_utc(self.clock())

    def day_for(self, moment: Optional[datetime] = None) -> date:
        return _as_utc(moment or self.now()).date()

    def path_for(self, day: date) -> Path:
        return self.root / (day_string(day) + ".db")

    def append(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError("daily memory stores only user and assistant messages")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("daily memory content must not be empty")
        moment = _as_utc(ts or self.now())
        day = moment.date()
        path = self.path_for(day)
        stamp = moment.isoformat()
        with self._open(path) as conn:
            cur = conn.execute(
                "INSERT INTO messages(ts, session_id, role, content) VALUES (?, ?, ?, ?)",
                (stamp, session_id or self.session_id, role, content),
            )
            return int(cur.lastrowid)

    def search(self, query: str, k: int = 8, days: int = 2) -> List[DailyHit]:
        tokens = _tokens(query)[:16]
        if not tokens:
            return []
        limit = max(1, min(int(k), 50))
        today = self.day_for()
        wanted: List[date] = []
        for offset in range(max(1, min(int(days), 366))):
            wanted.append(today - timedelta(days=offset))
        hits: List[DailyHit] = []
        for day in wanted:
            path = self.path_for(day)
            if not path.exists():
                continue
            hits.extend(self._search_day(day, tokens, limit))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _search_day(
        self, day: date, tokens: Sequence[str], limit: int
    ) -> List[DailyHit]:
        path = self.path_for(day)
        with self._open(path) as conn:
            if self._has_fts(conn):
                expr = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens[:16])
                try:
                    rows = conn.execute(
                        """
                        SELECT m.id, m.ts, m.role, m.content, bm25(messages_fts) AS rank
                        FROM messages_fts
                        JOIN messages m ON m.id = messages_fts.rowid
                        WHERE messages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (expr, limit),
                    ).fetchall()
                    return [
                        DailyHit(
                            day=day,
                            message_id=int(row[0]),
                            ts=str(row[1]),
                            role=str(row[2]),
                            content=str(row[3]),
                            score=-float(row[4]),
                        )
                        for row in rows
                    ]
                except sqlite3.OperationalError:
                    pass
            return self._search_like(conn, day, tokens, limit)

    def _search_like(
        self,
        conn: sqlite3.Connection,
        day: date,
        tokens: Sequence[str],
        limit: int,
    ) -> List[DailyHit]:
        clauses = " AND ".join("lower(content) LIKE ?" for _ in tokens)
        params = ["%" + token.lower() + "%" for token in tokens]
        rows = conn.execute(
            f"""
            SELECT id, ts, role, content
            FROM messages
            WHERE {clauses}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [
            DailyHit(
                day=day,
                message_id=int(row[0]),
                ts=str(row[1]),
                role=str(row[2]),
                content=str(row[3]),
                score=float(len(tokens)),
            )
            for row in rows
        ]

    def pending_days(self, now: Optional[datetime] = None) -> List[date]:
        today = self.day_for(now)
        pending: List[date] = []
        for path in self.root.glob("*.db"):
            match = _DAY_RE.fullmatch(path.name)
            if not match:
                continue
            try:
                day = date.fromisoformat(match.group(1))
            except ValueError:
                continue
            if day < today:
                pending.append(day)
        return sorted(pending)

    def all_days(self) -> List[date]:
        days: List[date] = []
        for path in self.root.glob("*.db"):
            match = _DAY_RE.fullmatch(path.name)
            if match:
                try:
                    days.append(date.fromisoformat(match.group(1)))
                except ValueError:
                    pass
        return sorted(days)

    def read_day(self, day: date) -> List[dict]:
        path = self.path_for(day)
        if not path.exists():
            return []
        with self._open(path) as conn:
            rows = conn.execute(
                "SELECT id, ts, session_id, role, content FROM messages ORDER BY id"
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "ts": str(row[1]),
                "session_id": str(row[2]),
                "role": str(row[3]),
                "content": str(row[4]),
            }
            for row in rows
        ]

    def count(self, day: Optional[date] = None) -> int:
        path = self.path_for(day or self.day_for())
        if not path.exists():
            return 0
        with self._open(path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return int(row[0]) if row else 0

    def delete_day(self, day: date) -> bool:
        path = self.path_for(day)
        removed = False
        for candidate in (
            path,
            Path(str(path) + "-wal"),
            Path(str(path) + "-shm"),
        ):
            try:
                candidate.unlink()
                removed = True
            except FileNotFoundError:
                pass
        return removed

    @contextmanager
    def _open(self, path: Path):
        conn = self._connect(path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.executescript(_SCHEMA)
        if self.use_fts:
            try:
                conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError:
                self.use_fts = False
        return conn

    @staticmethod
    def _has_fts(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        return row is not None


def _tokens(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out
