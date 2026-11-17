"""Provenance checks for persistent identity and session-state edits.

Prompt text can say "only the user may request this", but that is not an
enforcement boundary.  These helpers enforce the narrowest deterministic rule:
an edit is only accepted when the current turn contains no tool output or
untrusted user content, unless the user's own message explicitly asks for an
identity/state update.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_IDENTITY_REQUEST_RE = re.compile(
    r"(?is)(?:"
    r"\bremember\s+(?:me|this\s+about\s+me|my\b|that\s+my\b)"
    r"|\bmy\s+name\s+is\b"
    r"|\bcall\s+me\b"
    r"|\bi(?:'m|\s+am)\b"
    r"|\bi\s+(?:live|work|like|prefer|love|hate)\b"
    r"|\b(?:update|save|change|edit|add\s+to|remove\s+from)\b.{0,60}"
    r"\b(?:user|profile|identity|soul|persona)\b"
    r"|\b(?:change|update)\b.{0,40}\b(?:voice|values|persona|soul)\b"
    r")"
)
_STATE_REQUEST_RE = re.compile(
    r"(?is)(?:\b(?:update|save|write|set)\b.{0,40}\bstate\b|\bstate\s*:)"
)
_UNTRUSTED_PREFIX = "<untrusted"


def _content(message: Any) -> str:
    return str(getattr(message, "content", "") or "")


def _role(message: Any) -> str:
    return str(getattr(message, "role", "") or "")


def _last_user_index(messages: Sequence[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _role(messages[index]) == "user":
            return index
    return None


def _current_turn(messages: Sequence[Any], last_user: int) -> list[Any]:
    """Return the user turn plus any tool/assistant messages after it.

    Adjacent user messages before the last one are included so an attachment
    turn cannot hide behind the actual user request.
    """
    start = 0
    for index in range(last_user - 1, -1, -1):
        if _role(messages[index]) == "assistant":
            start = index + 1
            break
    return list(messages[start:])


def _has_untrusted_or_tool(messages: Sequence[Any]) -> bool:
    for message in messages:
        role = _role(message)
        if role == "tool":
            return True
        if role == "user" and _content(message).lstrip().startswith(
            _UNTRUSTED_PREFIX
        ):
            return True
    return False


def last_user_message(messages: Sequence[Any]) -> str:
    index = _last_user_index(messages)
    return _content(messages[index]) if index is not None else ""


def identity_edit_allowed(messages: Sequence[Any]) -> bool:
    """Whether a model-emitted USER.md/SOUL.md edit may be applied."""
    last = _last_user_index(messages)
    if last is None:
        return False
    if _IDENTITY_REQUEST_RE.search(_content(messages[last])):
        return True
    return not _has_untrusted_or_tool(_current_turn(messages, last))


def state_edit_allowed(messages: Sequence[Any]) -> bool:
    """Whether a model-emitted session-state edit may be applied."""
    last = _last_user_index(messages)
    if last is None:
        return False
    if _STATE_REQUEST_RE.search(_content(messages[last])):
        return True
    return not _has_untrusted_or_tool(_current_turn(messages, last))
