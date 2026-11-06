"""Configuration for the optional memory subsystem."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class MemoryConfig:
    root: Path = field(default_factory=lambda: Path.home() / ".sora_agent")
    enabled: bool = True
    user_id: str = "sora"

    recall_top_k: int = 8
    recall_max_chars: int = 6000
    recall_retrieval_mode: str = "auto"

    sidecar_version: str = "2.139.1"
    sidecar_binary: Path | None = None
    http_addr: str = "127.0.0.1:0"
    grpc_addr: str = "127.0.0.1:0"
    token: str | None = None
    auto_download: bool = True
    download_base_url: str = (
        "https://github.com/liliang-cn/cortexdb/releases/download"
    )
    request_timeout: float = 10.0
    sidecar_start_timeout: float = 20.0

    dream_auto: bool = True
    dream_rounds: int = 2
    dream_hour_utc: int = 0
    dream_max_memories: int = 500
    dream_max_merges: int = 20
    dream_max_conflicts: int = 20
    dream_max_inferences: int = 20
    dream_max_updates: int = 300
    dream_edge_jaccard: float = 0.45
    dream_decay: float = 0.98
    dream_min_importance: float = 0.05
    dream_budget_seconds: float = 120.0
    dream_rules: tuple[dict[str, Any], ...] = ()

    backup_keep_daily: int = 14
    backup_keep_weekly: int = 8
    backup_keep_monthly: int = 12
    backup_keep_active: int = 4

    wake_model: str = ""
    wake_base_url: str = ""
    wake_api_key: str = ""
    wake_timeout: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())
        object.__setattr__(
            self,
            "sidecar_binary",
            Path(self.sidecar_binary).expanduser()
            if self.sidecar_binary is not None
            else None,
        )
        if self.dream_rounds < 1:
            object.__setattr__(self, "dream_rounds", 1)
        if not 0 <= self.dream_hour_utc <= 23:
            object.__setattr__(self, "dream_hour_utc", 0)
        mode = str(self.recall_retrieval_mode or "auto").strip().lower()
        if mode not in {"auto", "lexical", "graph", "ppr"}:
            mode = "auto"
        object.__setattr__(self, "recall_retrieval_mode", mode)
        if self.recall_top_k < 1:
            object.__setattr__(self, "recall_top_k", 1)
        if self.recall_top_k > 50:
            object.__setattr__(self, "recall_top_k", 50)
        if self.recall_max_chars < 200:
            object.__setattr__(self, "recall_max_chars", 200)
        if self.dream_decay <= 0 or self.dream_decay > 1:
            object.__setattr__(self, "dream_decay", 0.98)
        if not 0 <= self.dream_edge_jaccard <= 1:
            object.__setattr__(self, "dream_edge_jaccard", 0.45)
        if self.dream_budget_seconds <= 0:
            object.__setattr__(self, "dream_budget_seconds", 120.0)

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def db_path(self) -> Path:
        return self.memory_dir / "cortexdb.db"

    @property
    def daily_dir(self) -> Path:
        return self.memory_dir / "daily"

    @property
    def backup_dir(self) -> Path:
        return self.root / "memory backup"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def log_dir(self) -> Path:
        return self.memory_dir / "logs"

    @property
    def report_dir(self) -> Path:
        return self.memory_dir / "dream" / "reports"

    @property
    def lock_path(self) -> Path:
        return self.memory_dir / "dream" / "dream.lock"

    @property
    def token_path(self) -> Path:
        return self.memory_dir / "grpc.token"

    @property
    def sidecar_path(self) -> Path:
        if self.sidecar_binary is not None:
            return self.sidecar_binary
        suffix = ".exe" if os.name == "nt" else ""
        return self.bin_dir / f"cortexdb-grpc{suffix}"

    @property
    def sidecar_log_path(self) -> Path:
        return self.log_dir / "cortexdb-grpc.log"

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            for path in (self.root, self.memory_dir, self.daily_dir,
                         self.backup_dir, self.log_dir, self.report_dir):
                try:
                    path.chmod(0o700)
                except OSError:
                    pass

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Build a config from ``SORA_MEMORY_*`` variables only."""
        return cls.from_settings({})

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any] | None = None,
        *,
        root: str | Path | None = None,
        wake_model: str = "",
        wake_base_url: str = "",
        wake_api_key: str = "",
    ) -> MemoryConfig:
        """Build a config from the wizard's memory section.

        Settings provide the base values; environment variables still win,
        preserving the documented precedence.  ``from_env()`` is the same
        function with an empty settings mapping.
        """
        raw = settings if isinstance(settings, Mapping) else {}
        defaults = cls()

        configured_root = root or raw.get("root")
        env_root = os.getenv("SORA_MEMORY_ROOT")
        resolved_root = (
            Path(env_root).expanduser()
            if env_root
            else (
                Path(configured_root).expanduser()
                if configured_root
                else defaults.root
            )
        )

        sidecar = _pick_bool(raw, "sidecar", True)
        auto_download = _pick_bool(raw, "auto_download", True) and sidecar
        dream = _pick_bool(raw, "dream", True)
        wake = _pick_bool(raw, "wake", True)

        binary_value = raw.get("sidecar_binary")
        env_binary = os.getenv("SORA_MEMORY_SIDECAR_BIN")
        binary = env_binary or binary_value

        rules: tuple[dict[str, Any], ...] = ()
        raw_rules = os.getenv("SORA_MEMORY_RULES")
        if raw_rules:
            try:
                parsed = json.loads(raw_rules)
                if isinstance(parsed, list):
                    rules = tuple(
                        item for item in parsed if isinstance(item, dict)
                    )
            except (TypeError, ValueError):
                rules = ()

        if not wake:
            wake_model = wake_base_url = wake_api_key = ""

        return cls(
            root=resolved_root,
            enabled=_env_bool(
                "SORA_MEMORY", _pick_bool(raw, "enabled", True)
            ),
            user_id=os.getenv("SORA_MEMORY_USER_ID", "sora"),
            recall_top_k=_env_int(
                "SORA_MEMORY_RECALL_TOP_K",
                _pick_int(raw, "recall_top_k", 8),
            ),
            recall_max_chars=_env_int(
                "SORA_MEMORY_RECALL_MAX_CHARS",
                _pick_int(raw, "recall_max_chars", 6000),
            ),
            recall_retrieval_mode=os.getenv(
                "SORA_MEMORY_RECALL_MODE",
                _pick_str(raw, "retrieval_mode", "auto"),
            ),
            sidecar_version=os.getenv(
                "SORA_MEMORY_SIDECAR_VERSION",
                _pick_str(raw, "sidecar_version", "2.139.1"),
            ),
            sidecar_binary=Path(binary).expanduser() if binary else None,
            http_addr=os.getenv("SORA_MEMORY_HTTP_ADDR", "127.0.0.1:0"),
            grpc_addr=os.getenv("SORA_MEMORY_GRPC_ADDR", "127.0.0.1:0"),
            token=os.getenv("SORA_MEMORY_TOKEN") or None,
            auto_download=_env_bool(
                "SORA_MEMORY_AUTO_DOWNLOAD", auto_download
            ),
            download_base_url=os.getenv(
                "SORA_MEMORY_DOWNLOAD_BASE_URL",
                "https://github.com/liliang-cn/cortexdb/releases/download",
            ),
            request_timeout=_env_float("SORA_MEMORY_REQUEST_TIMEOUT", 10.0),
            sidecar_start_timeout=_env_float(
                "SORA_MEMORY_START_TIMEOUT",
                _pick_float(raw, "sidecar_start_timeout", 20.0),
            ),
            dream_auto=_env_bool("SORA_MEMORY_DREAM_AUTO", dream),
            dream_rounds=_env_int(
                "SORA_MEMORY_DREAM_ROUNDS",
                _pick_int(raw, "dream_rounds", 2),
            ),
            dream_hour_utc=_env_int(
                "SORA_MEMORY_DREAM_HOUR_UTC",
                _pick_int(raw, "dream_hour_utc", 0),
            ),
            dream_max_memories=_env_int("SORA_MEMORY_DREAM_MAX_MEMORIES", 500),
            dream_max_merges=_env_int("SORA_MEMORY_DREAM_MAX_MERGES", 20),
            dream_max_conflicts=_env_int("SORA_MEMORY_DREAM_MAX_CONFLICTS", 20),
            dream_max_inferences=_env_int("SORA_MEMORY_DREAM_MAX_INFERENCES", 20),
            dream_max_updates=_env_int("SORA_MEMORY_DREAM_MAX_UPDATES", 300),
            dream_edge_jaccard=_env_float("SORA_MEMORY_DREAM_EDGE", 0.45),
            dream_decay=_env_float("SORA_MEMORY_DREAM_DECAY", 0.98),
            dream_min_importance=_env_float("SORA_MEMORY_DREAM_MIN_IMPORTANCE", 0.05),
            dream_budget_seconds=_env_float("SORA_MEMORY_DREAM_BUDGET", 120.0),
            dream_rules=rules,
            backup_keep_daily=_env_int("SORA_MEMORY_BACKUP_DAILY", 14),
            backup_keep_weekly=_env_int("SORA_MEMORY_BACKUP_WEEKLY", 8),
            backup_keep_monthly=_env_int("SORA_MEMORY_BACKUP_MONTHLY", 12),
            backup_keep_active=_env_int("SORA_MEMORY_BACKUP_ACTIVE", 4),
            wake_model=os.getenv("SORA_MEMORY_WAKE_MODEL", wake_model),
            wake_base_url=os.getenv(
                "SORA_MEMORY_WAKE_BASE_URL", wake_base_url
            ),
            wake_api_key=os.getenv("SORA_MEMORY_WAKE_API_KEY", wake_api_key),
            wake_timeout=_env_float("SORA_MEMORY_WAKE_TIMEOUT", 60.0),
        )


def _pick_bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _pick_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = raw.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_str(raw: Mapping[str, Any], key: str, default: str) -> str:
    value = raw.get(key)
    return default if value is None or value == "" else str(value)
