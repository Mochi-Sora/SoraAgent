"""Best-effort append-only audit trail for automatic document edits.

Identity and session state change without human review, so there must be a
durable record of *when* they changed and *what* the old and new versions
hashed to.  Audit failures never propagate: losing a log line must not block
the agent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON object to ``path`` as a single line.

    The file is opened with ``O_APPEND`` and written once, so concurrent
    writers do not interleave partial records on POSIX filesystems.
    """
    record = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **event,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        try:
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError):
        return
