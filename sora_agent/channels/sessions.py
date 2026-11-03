"""Durable per-session conversation history.

One JSON file per session, written atomically and readable on startup.  The
store is deliberately independent of the memory subsystem: memory is optional
and summarizes, while this file is the exact conversation the model replays.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..providers import Message, ToolCall

log = logging.getLogger(__name__)

_VERSION = 1


def default_root() -> Path:
    configured = os.getenv("SORA_SESSIONS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".sora_agent" / "sessions"


class SessionStore:
    """Load and save one conversation per session key."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.enabled = enabled

    def path(self, session_key: str) -> Path:
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
        return self.root / f"{digest}.json"

    def load(self, session_key: str) -> list[Message]:
        if not self.enabled:
            return []
        try:
            raw = self.path(session_key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            log.warning("session %s could not be read: %s", session_key, exc)
            return []
        try:
            record = json.loads(raw)
            if not isinstance(record, dict) or record.get("version") != _VERSION:
                return []
            items = record.get("messages")
            if not isinstance(items, list):
                return []
            return [_message_from_dict(item) for item in items]
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("session %s is corrupt: %s", session_key, exc)
            return []

    def save(self, session_key: str, messages: list[Message]) -> None:
        if not self.enabled or not messages:
            return
        record = {
            "version": _VERSION,
            "session_key": session_key,
            "messages": [_message_to_dict(item) for item in messages],
        }
        text = json.dumps(record, ensure_ascii=False, indent=2, default=str)
        self._atomic_write(self.path(session_key), text)

    def clear(self, session_key: str) -> None:
        if not self.enabled:
            return
        try:
            self.path(session_key).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("session %s could not be cleared: %s", session_key, exc)

    def _atomic_write(self, target: Path, text: str) -> None:
        root_existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        if not root_existed:
            try:
                os.chmod(self.root, 0o700)
            except OSError:
                pass

        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(self.root),
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            os.replace(temporary_path, target)
        except OSError as exc:
            log.warning("session write failed: %s", exc)
        finally:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "raw_arguments": call.raw_arguments,
                "error": call.error,
            }
            for call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "name": message.name,
    }


def _message_from_dict(item: Any) -> Message:
    if not isinstance(item, dict):
        raise TypeError("session message must be an object")
    role = item.get("role")
    if not isinstance(role, str):
        raise TypeError("session message role must be a string")
    calls = []
    for raw in item.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        arguments = raw.get("arguments")
        calls.append(
            ToolCall(
                id=str(raw.get("id") or ""),
                name=str(raw.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                raw_arguments=str(raw.get("raw_arguments") or ""),
                error=str(raw.get("error") or ""),
            )
        )
    return Message(
        role=role,
        content=str(item.get("content") or ""),
        tool_calls=calls,
        tool_call_id=str(item.get("tool_call_id") or ""),
        name=str(item.get("name") or ""),
    )
