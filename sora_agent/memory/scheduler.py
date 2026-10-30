"""Process-local scheduler for automatic nightly consolidation.

The scheduler is intentionally small and dependency-free: a daemon thread
sleeps until the next configured UTC hour, then calls ``MemoryHook.dream``.
If the process starts after the configured hour and daily files are still
waiting, it runs one catch-up dream before settling into the nightly cadence.
Processes that are not alive at midnight are covered by that catch-up path.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from datetime import time as clock_time
from typing import Any, Callable

from .config import MemoryConfig


def scheduled_for(day: datetime, hour: int) -> datetime:
    """Return the configured UTC wall-clock time on ``day``'s date."""

    return datetime.combine(
        day.date(),
        clock_time(hour=max(0, min(23, int(hour)))),
        tzinfo=timezone.utc,
    )


def next_dream_at(now: datetime, hour: int) -> datetime:
    """Return the next scheduled dream time strictly after ``now``."""

    moment = _as_utc(now)
    target = scheduled_for(moment, hour)
    if moment < target:
        return target
    return target + timedelta(days=1)


class DreamScheduler:
    """Run ``dream`` once per configured midnight in a daemon thread."""

    def __init__(
        self,
        dream: Callable[[], Any],
        config: MemoryConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        has_pending: Callable[[datetime], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.dream = dream
        self.config = config
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.has_pending = has_pending
        self.log = logger or logging.getLogger("sora_agent.memory")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="sora-memory-dream",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)
        self._thread = None

    def _loop(self) -> None:
        now = _as_utc(self.clock())
        if self._catch_up_due(now):
            self._run()
        while not self._stop.is_set():
            now = _as_utc(self.clock())
            target = next_dream_at(now, self.config.dream_hour_utc)
            delay = max(0.0, (target - now).total_seconds())
            if self._stop.wait(delay):
                return
            self._run()

    def _catch_up_due(self, now: datetime) -> bool:
        if self.has_pending is None:
            return False
        if now < scheduled_for(now, self.config.dream_hour_utc):
            return False
        try:
            return bool(self.has_pending(now))
        except Exception as exc:  # noqa: BLE001 - scheduling is best effort
            self.log.warning("dream catch-up check failed: %s", exc)
            return False

    def _run(self) -> None:
        if self._stop.is_set():
            return
        try:
            self.dream()
        except Exception as exc:  # noqa: BLE001 - keep the scheduler alive
            self.log.warning("automatic dream failed: %s", exc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
