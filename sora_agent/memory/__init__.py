"""CortexDB-backed memory for SoraAgent.

The core modules here use only the standard library and know nothing about the
agent loop. ``sora_agent.memory.hook`` is the single optional adapter.
"""

from __future__ import annotations

from typing import Any

from .backup import BackupError, BackupManager
from .client import CortexDBClient, CortexDBError, CortexDBUnavailable
from .config import MemoryConfig
from .daily import DailyHit, DailyStore
from .dream import DreamBusy, DreamError, DreamRunner
from .gate import RecallEnforcingProvider
from .louvain import louvain_communities
from .maintenance import optimize_active
from .recall import MemoryRecallService, RecallItem
from .sidecar import SidecarError, SidecarManager

__all__ = [
    "BackupError",
    "BackupManager",
    "CortexDBClient",
    "CortexDBError",
    "CortexDBUnavailable",
    "DailyHit",
    "DailyStore",
    "DreamBusy",
    "DreamError",
    "DreamRunner",
    "MemoryConfig",
    "MemoryRecallService",
    "RecallEnforcingProvider",
    "RecallItem",
    "SidecarError",
    "SidecarManager",
    "louvain_communities",
    "optimize_active",
]


def __getattr__(name: str) -> Any:
    if name in {"MemoryHook", "default_memory"}:
        from . import hook

        return getattr(hook, name)
    raise AttributeError(name)
