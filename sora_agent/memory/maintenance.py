"""Offline SQLite maintenance for the active memory database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import MemoryConfig


def optimize_active(config: MemoryConfig, vacuum: bool = False) -> dict[str, Any]:
    path = Path(config.db_path)
    if not path.exists():
        return {"ok": False, "reason": "database does not exist", "path": str(path)}
    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA optimize")
        conn.execute("ANALYZE")
        if vacuum:
            conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()
    return {
        "ok": True,
        "path": str(path),
        "vacuumed": bool(vacuum),
        "bytes": path.stat().st_size,
    }
