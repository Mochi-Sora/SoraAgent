"""Small, dependency-free helpers shared by channel packages."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def split_ids(raw: str) -> tuple[str, ...]:
    """Split a comma-separated id list, dropping blanks."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value, default: bool) -> bool:
    if value is None or not str(value).strip():
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return default


def env_str(
    name: str,
    default: str = "",
    env: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if env is None else env
    value = source.get(name)
    return default if value is None or not value.strip() else value.strip()


def env_int(
    name: str,
    default: int,
    env: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if env is None else env
    return as_int(source.get(name), default)


def env_float(
    name: str,
    default: float,
    env: Mapping[str, str] | None = None,
) -> float:
    source = os.environ if env is None else env
    return as_float(source.get(name), default)


def env_bool(
    name: str,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if env is None else env
    return as_bool(source.get(name), default)


def split_text(text: str, limit: int) -> list[str]:
    """Split text into chunks no longer than ``limit`` characters."""
    if limit <= 0:
        raise ValueError("limit must be > 0")
    remaining = (text or "").strip()
    chunks: list[str] = []
    while len(remaining) > limit:
        window = remaining[:limit]
        if remaining[limit] in " \n\t\r":
            cut = limit
        else:
            cut = window.rfind("\n")
            if cut < limit // 2:
                cut = window.rfind(" ")
            if cut < limit // 2:
                cut = limit
        chunk = remaining[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def now_ms() -> int:
    return int(time.time() * 1000)


class Dedupe:
    """A bounded, thread-safe set of recently seen platform message ids."""

    def __init__(self, maxsize: int = 512) -> None:
        self._maxsize = max(1, int(maxsize))
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def seen(self, key: str | None) -> bool:
        """Return True when ``key`` was already recorded."""
        if not key:
            return False
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return True
            self._seen[key] = None
            if len(self._seen) > self._maxsize:
                self._seen.popitem(last=False)
            return False
