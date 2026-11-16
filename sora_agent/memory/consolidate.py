"""Simple daily-store consolidation for daily-only memory.

No transfer, relationships, facts, conflicts, communities, decay, or wake
gates.  It only removes exact duplicates and compacts the SQLite files.  The
previous snapshot is compressed into an ``archive/`` folder beside the daily
store before anything changes.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable

from .backup import BackupManager
from .config import MemoryConfig
from .daily import DailyStore
from .specialists import SpecialistSource


class ConsolidationError(RuntimeError):
    pass


class ConsolidationBusy(ConsolidationError):
    pass


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


class DailyConsolidator:
    def __init__(
        self,
        config: MemoryConfig,
        daily: DailyStore,
        backup: BackupManager | None = None,
        *,
        specialists: list[SpecialistSource] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.daily = daily
        self.backup = backup or BackupManager(config, clock=clock)
        self.specialists = list(specialists or [])
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.started = 0.0

    def now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def run(self, dry_run: bool = False, include_today: bool = False) -> dict[str, Any]:
        self.started = time.monotonic()
        now = self.now()
        report: dict[str, Any] = {
            "mode": "daily_only",
            "started_at": now.isoformat(),
            "finished_at": "",
            "dry_run": dry_run,
            "consolidated_days": [],
            "errors": [],
        }
        with self._lock():
            for label, store, agent_id in self._stores():
                for day in self._pending(store, now, include_today):
                    self._consolidate_day(
                        report, label, store, agent_id, day, dry_run
                    )
        report["finished_at"] = self.now().isoformat()
        report["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        return report

    def _stores(self):
        yield "main", self.daily, ""
        for source in self.specialists:
            yield f"specialist:{source.agent_id}", source.store, source.agent_id

    def _pending(
        self, store: DailyStore, now: datetime, include_today: bool
    ) -> list:
        pending = store.pending_days(now)
        if include_today and store.count() > 0:
            today = store.day_for(now)
            if today not in pending:
                pending.append(today)
        return sorted(pending)

    def _consolidate_day(
        self,
        report: dict[str, Any],
        label: str,
        store: DailyStore,
        agent_id: str,
        day,
        dry_run: bool,
    ) -> None:
        messages = store.read_day(day)
        if not messages:
            return
        if not dry_run:
            try:
                archive = store.root / "archive"
                archive.mkdir(parents=True, exist_ok=True)
                self.backup.snapshot(
                    store.path_for(day),
                    "daily",
                    day.isoformat(),
                    directory=archive,
                )
            except Exception as exc:  # noqa: BLE001 - one backup failure is data
                report["errors"].append(f"{label} backup {day}: {exc}")
                return
        seen: set[str] = set()
        duplicates: list[int] = []
        for message in messages:
            normalized = _normalize(message.get("content", ""))
            if not normalized:
                duplicates.append(int(message["id"]))
                continue
            if normalized in seen:
                duplicates.append(int(message["id"]))
            else:
                seen.add(normalized)
        report["consolidated_days"].append(
            {
                "source": label,
                "day": day.isoformat(),
                "before": len(messages),
                "duplicates": len(duplicates),
                "deleted": 0 if dry_run else len(duplicates),
            }
        )
        if dry_run or not duplicates:
            return
        try:
            store.delete_messages(day, duplicates)
            store.compact(day)
            report["consolidated_days"][-1]["deleted"] = len(duplicates)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{label} consolidate {day}: {exc}")

    @contextmanager
    def _lock(self):
        self.config.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.lock_path, "a+") as handle:
            try:
                if os.name == "nt":
                    import msvcrt

                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError as exc:
                        raise ConsolidationBusy("another memory job is running") from exc
                else:
                    import fcntl

                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as exc:
                        raise ConsolidationBusy("another memory job is running") from exc
                yield
            finally:
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
