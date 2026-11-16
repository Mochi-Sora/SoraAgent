"""Per-specialist daily stores backed by the shared SQLite DailyStore.

Long-term memory is unified; specialists only own ephemeral daily rows.  Using
the same DailyStore implementation as the main agent gives dreaming stable row
IDs, WAL concurrency, pending-day discovery, and row-level deletion.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from ..memory.daily import DailyStore


class SpecialistDaily:
    def __init__(self, workspace: Path, agent_id: str) -> None:
        self.workspace = Path(workspace)
        self.agent_id = str(agent_id)
        self.store = DailyStore(
            self.workspace / "daily",
            session_id=f"specialist:{self.agent_id}",
        )
        self.root = self.store.root

    def remember(self, text: str) -> str:
        message_id = self.store.append("assistant", text)
        return f"Stored in today's daily log (row {message_id})."

    def recall(self, query: str = "", *, days: int = 7, k: int = 20) -> str:
        if str(query or "").strip():
            hits = self.store.search(query, k=k, days=days)
            if not hits:
                return f"No daily entries matched {query!r}."
            return "\n".join(
                f"- [{hit.day.isoformat()} {hit.ts[11:16]}] {hit.content}"
                for hit in hits
            )
        today = self.store.day_for()
        remaining = max(1, min(int(k), 100))
        lines: list[str] = []
        for offset in range(max(1, min(int(days), 366))):
            day = today - timedelta(days=offset)
            for row in reversed(self.store.read_day(day)):
                lines.append(
                    f"- [{day.isoformat()} {str(row['ts'])[11:16]}] {row['content']}"
                )
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
        return "\n".join(lines) if lines else f"No daily entries in the last {days} day(s)."

    def pending_days(self, now: datetime | None = None) -> list[date]:
        return self.store.pending_days(now)
