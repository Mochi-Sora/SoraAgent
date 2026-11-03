"""Model-facing clock.

The agent loop has no sense of "now": the prompt is static, and inbound
timestamps are transport metadata rather than model context.  This module
appends one small, per-turn ``## CURRENT TIME`` section to the system prompt so
the model can resolve "today", "tomorrow", and relative deadlines.

The section is appended *after* every other system section.  The identity and
state wrappers use :func:`extract_clock` / :func:`append_clock` to keep that
ordering even when wrappers are composed in a different order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from .providers import Message, ModelTurn, provider_text, provider_turn

log = logging.getLogger(__name__)

CLOCK_HEADING = "## CURRENT TIME"
_CLOCK_SEPARATOR = "\n\n" + CLOCK_HEADING


def default_clock() -> datetime:
    """Return the current local, timezone-aware time."""
    return datetime.now().astimezone()


def clock_section(now: datetime | None = None) -> str:
    """Render the clock section for one moment.

    ``now`` may be naive for tests; a naive value is interpreted as local time,
    matching :func:`default_clock`.
    """
    moment = now or default_clock()
    if moment.tzinfo is None:
        moment = moment.astimezone()

    local = moment.isoformat(timespec="seconds")
    weekday = moment.strftime("%A")
    zone = moment.tzname() or moment.strftime("UTC%z")
    utc = moment.astimezone(timezone.utc)
    return (
        f"{CLOCK_HEADING}\n"
        f"Local: {local} ({weekday}, {zone})\n"
        f"UTC: {utc.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        "Treat this as the authoritative current time. Resolve relative dates "
        'such as "today", "tomorrow", and "next week" against it.'
    )


def is_clock_section(content: str) -> bool:
    """Whether a system message is a clock section (possibly embedded)."""
    text = str(content or "")
    return text == CLOCK_HEADING or text.startswith(CLOCK_HEADING + "\n")


def extract_clock(messages: Sequence[Any]) -> str:
    """Return the last clock section found in ``messages``, or ``""``."""
    found = ""
    for message in messages:
        if getattr(message, "role", "") != "system":
            continue
        content = str(getattr(message, "content", "") or "")
        if is_clock_section(content):
            found = content
            continue
        marker = content.find(_CLOCK_SEPARATOR)
        if marker >= 0:
            found = content[marker + 2 :]
    return found


def strip_clock(messages: Sequence[Any]) -> list[Any]:
    """Return ``messages`` without clock sections.

    A dedicated clock message is removed.  If a clock section was appended to
    another system message, only that trailing section is removed.
    """
    cleaned: list[Any] = []
    for message in messages:
        if getattr(message, "role", "") != "system":
            cleaned.append(message)
            continue
        content = str(getattr(message, "content", "") or "")
        if is_clock_section(content):
            continue
        marker = content.find(_CLOCK_SEPARATOR)
        if marker >= 0:
            text = content[:marker].rstrip()
            if text:
                cleaned.append(Message("system", text))
            continue
        cleaned.append(message)
    return cleaned


def append_clock(messages: Sequence[Any], section: str) -> list[Any]:
    """Append a clock section after the leading system-message block."""
    if not section:
        return list(messages)
    augmented = list(messages)
    insert_at = 0
    for index, message in enumerate(augmented):
        if getattr(message, "role", "") != "system":
            break
        insert_at = index + 1
    augmented.insert(insert_at, Message("system", section))
    return augmented


def with_clock(messages: Sequence[Any], now: datetime | None = None) -> list[Any]:
    """Replace any existing clock and append a fresh section at the end."""
    return append_clock(strip_clock(messages), clock_section(now))


def resolve_clock(
    clock: Any = None,
) -> Callable[[], datetime] | None:
    """Turn a clock option into a callable, or ``None`` when disabled.

    ``None``/``True``/``"default"`` select the wall clock, a ``datetime`` is a
    fixed clock (useful in tests), and a callable is used as-is.
    """
    if clock is False:
        return None
    if clock is None or clock is True or clock == "default":
        return default_clock
    if isinstance(clock, datetime):
        fixed = clock
        return lambda: fixed
    if callable(clock):
        return cast(Callable[[], datetime], clock)
    raise TypeError(
        "clock must be False, a datetime, or a callable returning a datetime"
    )


class ClockProvider:
    """Wrap a provider and inject the current time on every turn."""

    __sora_clock__ = True

    def __init__(self, inner: Any, clock: Any = None) -> None:
        self.inner = inner
        self.clock = resolve_clock(clock) or default_clock
        # Keep identity double-wrap detection working through a clock wrapper.
        self.__sora_identity__ = getattr(inner, "__sora_identity__", False)

    def complete_turn(
        self, messages: list[Message], tools: list[Any]
    ) -> ModelTurn:
        augmented = self._augment(messages)
        return provider_turn(self.inner, augmented, tools)

    def complete(self, messages: list[Message], tools: list[Any]) -> str:
        augmented = self._augment(messages)
        return provider_text(self.inner, augmented, tools)

    def stream_turn(
        self,
        messages: list[Message],
        tools: list[Any],
        on_delta: Callable[[str], None],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ModelTurn:
        augmented = self._augment(messages)
        stream = getattr(self.inner, "stream_turn", None)
        if callable(stream):
            return cast(
                ModelTurn,
                stream(augmented, tools, on_delta, should_stop=should_stop),
            )
        turn = provider_turn(self.inner, augmented, tools)
        if turn.text:
            on_delta(turn.text)
        return turn

    def _augment(self, messages: Sequence[Any]) -> list[Any]:
        try:
            return with_clock(messages, self.clock())
        except Exception as exc:  # noqa: BLE001 - a broken clock must fail open
            log.warning("clock injection failed: %s", exc)
            return list(messages)


def wrap_clock(provider: Any, clock: Any = None) -> Any:
    """Wrap ``provider`` with clock injection, avoiding double wrapping."""
    if clock is False or getattr(provider, "__sora_clock__", False):
        return provider
    try:
        resolved = resolve_clock(clock)
    except TypeError as exc:
        log.warning("clock disabled: %s", exc)
        return provider
    if resolved is None:
        return provider
    return ClockProvider(provider, resolved)


__all__ = [
    "CLOCK_HEADING",
    "ClockProvider",
    "append_clock",
    "clock_section",
    "default_clock",
    "extract_clock",
    "is_clock_section",
    "resolve_clock",
    "strip_clock",
    "with_clock",
    "wrap_clock",
]
