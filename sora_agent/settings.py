"""Persistent settings for the ``sora`` CLI and the gateway.

The config file holds no secrets.  It is a plain JSON document at
``~/.sora_agent/config.json``; API keys and bot tokens live in
``~/.sora_agent/secrets.env``.

Precedence is always:

    command-line flag  >  environment variable  >  config file  >  built-in
    default

Everything here is standard-library-only.  The Typer/Rich/Questionary wizard
lives in :mod:`sora_agent.cli` and imports this module, never the other way
around.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

CONFIG_VERSION = 1
DEFAULT_ROOT = Path.home() / ".sora_agent"
DEFAULT_CONFIG_PATH = DEFAULT_ROOT / "config.json"
DEFAULT_SECRETS_PATH = DEFAULT_ROOT / "secrets.env"
CONFIG_PATH_ENV = "SORA_CONFIG"
SECRETS_PATH_ENV = "SORA_SECRETS"

MODES = ("recommended", "advanced")
PERSONALITY_BASES = (
    "professional",
    "friendly",
    "candid",
    "efficient",
    "neutral",
)
WARMTH_LEVELS = ("very_cold", "cold", "neutral", "warm", "very_warm")
APPROVAL_MODES = ("off", "ask")
TOOL_SETS = ("none", "demo")


class SettingsError(ValueError):
    """Raised when a settings file is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Small, tolerant parsers
# ---------------------------------------------------------------------------


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SettingsError(f"{label} must be a JSON object")
    return value


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"expected a boolean, got {value!r}")


def _as_int(
    value: Any, default: int, *, minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"expected an integer, got {value!r}") from exc
    if minimum is not None and number < minimum:
        raise SettingsError(f"{number} is below the minimum of {minimum}")
    if maximum is not None and number > maximum:
        raise SettingsError(f"{number} is above the maximum of {maximum}")
    return number


def _as_float(
    value: Any, default: float, *, minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"expected a number, got {value!r}") from exc
    if minimum is not None and number < minimum:
        raise SettingsError(f"{number} is below the minimum of {minimum}")
    if maximum is not None and number > maximum:
        raise SettingsError(f"{number} is above the maximum of {maximum}")
    return number


def _as_str(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _as_optional_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text or None


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise SettingsError("expected a list of strings")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class PersonalitySettings:
    base: str = "friendly"
    warmth: str = "very_warm"

    def __post_init__(self) -> None:
        if self.base not in PERSONALITY_BASES:
            raise SettingsError(
                f"personality.base must be one of {', '.join(PERSONALITY_BASES)}"
            )
        if self.warmth not in WARMTH_LEVELS:
            raise SettingsError(
                f"personality.warmth must be one of {', '.join(WARMTH_LEVELS)}"
            )

    @property
    def enabled(self) -> bool:
        return self.base != "neutral" or self.warmth != "neutral"

    @classmethod
    def from_dict(cls, data: Any) -> PersonalitySettings:
        raw = _mapping(data, "agent.personality")
        return cls(
            base=_as_str(raw.get("base"), "friendly"),
            warmth=_as_str(raw.get("warmth"), "very_warm"),
        )


@dataclass
class AgentSettings:
    name: str = "Sora"
    icon: str = "☁️"
    personality: PersonalitySettings = field(default_factory=PersonalitySettings)

    def __post_init__(self) -> None:
        self.name = self.name.strip() or "Sora"
        self.icon = self.icon.strip()

    @classmethod
    def from_dict(cls, data: Any) -> AgentSettings:
        raw = _mapping(data, "agent")
        return cls(
            name=_as_str(raw.get("name"), "Sora"),
            icon=_as_str(raw.get("icon"), "☁️"),
            personality=PersonalitySettings.from_dict(raw.get("personality")),
        )


@dataclass
class ProviderSettings:
    kind: str = "openai"
    model: str = "gpt-5.4-mini"
    temperature: float = 0.7
    context_window: int = 272_000
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.kind = self.kind.strip().lower() or "openai"
        self.temperature = _as_float(
            self.temperature, 0.7, minimum=0.0, maximum=2.0
        )
        self.context_window = _as_int(
            self.context_window, 272_000, minimum=1_000
        )
        self.base_url = _as_optional_str(self.base_url)

    @classmethod
    def from_dict(cls, data: Any) -> ProviderSettings:
        raw = _mapping(data, "provider")
        return cls(
            kind=_as_str(raw.get("kind"), "openai"),
            model=_as_str(raw.get("model"), "gpt-5.4-mini"),
            temperature=_as_float(raw.get("temperature"), 0.7),
            context_window=_as_int(raw.get("context_window"), 272_000),
            base_url=_as_optional_str(raw.get("base_url")),
        )


@dataclass
class MemorySettings:
    enabled: bool = True
    sidecar: bool = True
    auto_download: bool = True
    dream: bool = True
    backup: bool = True
    wake: bool = True
    recall_top_k: int = 8
    recall_max_chars: int = 6000
    retrieval_mode: str = "auto"
    dream_hour_utc: int = 0
    dream_rounds: int = 2
    sidecar_version: str = "2.139.1"
    sidecar_binary: str | None = None
    sidecar_start_timeout: float = 20.0

    def __post_init__(self) -> None:
        self.recall_top_k = _as_int(
            self.recall_top_k, 8, minimum=1, maximum=50
        )
        self.recall_max_chars = _as_int(
            self.recall_max_chars, 6000, minimum=200
        )
        mode = str(self.retrieval_mode or "auto").strip().lower()
        if mode not in {"auto", "lexical", "graph", "ppr"}:
            mode = "auto"
        self.retrieval_mode = mode
        self.dream_hour_utc = _as_int(
            self.dream_hour_utc, 0, minimum=0, maximum=23
        )
        self.dream_rounds = _as_int(
            self.dream_rounds, 2, minimum=1, maximum=20
        )
        self.sidecar_start_timeout = _as_float(
            self.sidecar_start_timeout, 20.0, minimum=1.0
        )
        self.sidecar_binary = _as_optional_str(self.sidecar_binary)

    @classmethod
    def from_dict(cls, data: Any) -> MemorySettings:
        raw = _mapping(data, "features.memory")
        base = cls()
        return cls(
            enabled=_as_bool(raw.get("enabled"), base.enabled),
            sidecar=_as_bool(raw.get("sidecar"), base.sidecar),
            auto_download=_as_bool(
                raw.get("auto_download"), base.auto_download
            ),
            dream=_as_bool(raw.get("dream"), base.dream),
            backup=_as_bool(raw.get("backup"), base.backup),
            wake=_as_bool(raw.get("wake"), base.wake),
            recall_top_k=_as_int(raw.get("recall_top_k"), base.recall_top_k),
            recall_max_chars=_as_int(
                raw.get("recall_max_chars"), base.recall_max_chars
            ),
            retrieval_mode=_as_str(
                raw.get("retrieval_mode"), base.retrieval_mode
            ),
            dream_hour_utc=_as_int(
                raw.get("dream_hour_utc"), base.dream_hour_utc
            ),
            dream_rounds=_as_int(
                raw.get("dream_rounds"), base.dream_rounds
            ),
            sidecar_version=_as_str(
                raw.get("sidecar_version"), base.sidecar_version
            ),
            sidecar_binary=_as_optional_str(raw.get("sidecar_binary")),
            sidecar_start_timeout=_as_float(
                raw.get("sidecar_start_timeout"), base.sidecar_start_timeout
            ),
        )


@dataclass
class FeatureSettings:
    memory: MemorySettings = field(default_factory=MemorySettings)
    identity: bool = True
    clock: bool = True
    calendar: bool = True
    sessions: bool = True
    tools: str = "demo"
    streaming: bool = True
    approval: str = "ask"
    quality: bool = True

    def __post_init__(self) -> None:
        tools = str(self.tools or "none").strip().lower()
        if tools not in TOOL_SETS:
            raise SettingsError(f"features.tools must be one of {TOOL_SETS}")
        self.tools = tools
        approval = str(self.approval or "ask").strip().lower()
        if approval not in APPROVAL_MODES:
            raise SettingsError(
                f"features.approval must be one of {APPROVAL_MODES}"
            )
        self.approval = approval

    @classmethod
    def from_dict(cls, data: Any) -> FeatureSettings:
        raw = _mapping(data, "features")
        base = cls()
        return cls(
            memory=MemorySettings.from_dict(raw.get("memory")),
            identity=_as_bool(raw.get("identity"), base.identity),
            clock=_as_bool(raw.get("clock"), base.clock),
            calendar=_as_bool(raw.get("calendar"), base.calendar),
            sessions=_as_bool(raw.get("sessions"), base.sessions),
            tools=_as_str(raw.get("tools"), base.tools),
            streaming=_as_bool(raw.get("streaming"), base.streaming),
            approval=_as_str(raw.get("approval"), base.approval),
            quality=_as_bool(raw.get("quality"), base.quality),
        )


@dataclass
class LimitSettings:
    max_steps: int = 12
    max_turns: int = 80
    provider_timeout: float = 60.0
    provider_retries: int = 3
    tool_timeout: float = 60.0
    approval_timeout: float = 300.0
    max_tool_result_chars: int = 4000
    max_stream_chars: int = 3500

    def __post_init__(self) -> None:
        self.max_steps = _as_int(self.max_steps, 12, minimum=1)
        self.max_turns = _as_int(self.max_turns, 80, minimum=0)
        self.provider_timeout = _as_float(
            self.provider_timeout, 60.0, minimum=1.0
        )
        self.provider_retries = _as_int(
            self.provider_retries, 3, minimum=0, maximum=20
        )
        self.tool_timeout = _as_float(self.tool_timeout, 60.0, minimum=0.1)
        self.approval_timeout = _as_float(
            self.approval_timeout, 300.0, minimum=1.0
        )
        self.max_tool_result_chars = _as_int(
            self.max_tool_result_chars, 4000, minimum=0
        )
        self.max_stream_chars = _as_int(
            self.max_stream_chars, 3500, minimum=1
        )

    @classmethod
    def from_dict(cls, data: Any) -> LimitSettings:
        raw = _mapping(data, "limits")
        base = cls()
        return cls(
            max_steps=_as_int(raw.get("max_steps"), base.max_steps),
            max_turns=_as_int(raw.get("max_turns"), base.max_turns),
            provider_timeout=_as_float(
                raw.get("provider_timeout"), base.provider_timeout
            ),
            provider_retries=_as_int(
                raw.get("provider_retries"), base.provider_retries
            ),
            tool_timeout=_as_float(
                raw.get("tool_timeout"), base.tool_timeout
            ),
            approval_timeout=_as_float(
                raw.get("approval_timeout"), base.approval_timeout
            ),
            max_tool_result_chars=_as_int(
                raw.get("max_tool_result_chars"), base.max_tool_result_chars
            ),
            max_stream_chars=_as_int(
                raw.get("max_stream_chars"), base.max_stream_chars
            ),
        )


@dataclass
class ChannelSettings:
    kind: str = "telegram"
    allow_from: tuple[str, ...] = ()
    parse_mode: str | None = None
    api_base: str = "https://api.telegram.org"
    poll_timeout: int = 30
    request_timeout: float = 40.0
    max_message_chars: int = 4096

    def __post_init__(self) -> None:
        self.kind = self.kind.strip().lower() or "telegram"
        self.api_base = self.api_base.strip() or "https://api.telegram.org"
        self.poll_timeout = _as_int(self.poll_timeout, 30, minimum=0)
        self.request_timeout = _as_float(
            self.request_timeout, 40.0, minimum=1.0
        )
        self.max_message_chars = _as_int(
            self.max_message_chars, 4096, minimum=1
        )
        parse_mode = _as_optional_str(self.parse_mode)
        if parse_mode is not None and parse_mode not in {
            "HTML", "Markdown", "MarkdownV2"
        }:
            raise SettingsError(
                "channel.parse_mode must be HTML, Markdown, or MarkdownV2"
            )
        self.parse_mode = parse_mode

    @classmethod
    def from_dict(cls, data: Any) -> ChannelSettings:
        raw = _mapping(data, "channel")
        base = cls()
        return cls(
            kind=_as_str(raw.get("kind"), base.kind),
            allow_from=_as_str_tuple(raw.get("allow_from")),
            parse_mode=_as_optional_str(raw.get("parse_mode")),
            api_base=_as_str(raw.get("api_base"), base.api_base),
            poll_timeout=_as_int(
                raw.get("poll_timeout"), base.poll_timeout
            ),
            request_timeout=_as_float(
                raw.get("request_timeout"), base.request_timeout
            ),
            max_message_chars=_as_int(
                raw.get("max_message_chars"), base.max_message_chars
            ),
        )


@dataclass
class PathSettings:
    sessions_root: str | None = None
    calendar_db: str | None = None
    identity_root: str | None = None
    memory_root: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> PathSettings:
        raw = _mapping(data, "paths")
        return cls(
            sessions_root=_as_optional_str(raw.get("sessions_root")),
            calendar_db=_as_optional_str(raw.get("calendar_db")),
            identity_root=_as_optional_str(raw.get("identity_root")),
            memory_root=_as_optional_str(raw.get("memory_root")),
        )

    def as_path(self, value: str | None) -> Path | None:
        return Path(value).expanduser() if value else None


@dataclass
class SetupSettings:
    mode: str = "recommended"
    completed_at: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES and self.mode != "legacy":
            raise SettingsError(f"setup.mode must be one of {MODES}")

    @classmethod
    def from_dict(cls, data: Any) -> SetupSettings:
        raw = _mapping(data, "setup")
        return cls(
            mode=_as_str(raw.get("mode"), "recommended"),
            completed_at=_as_str(raw.get("completed_at"), ""),
        )


@dataclass
class Settings:
    version: int = CONFIG_VERSION
    setup: SetupSettings = field(default_factory=SetupSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    channel: ChannelSettings = field(default_factory=ChannelSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    limits: LimitSettings = field(default_factory=LimitSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Settings:
        raw = _mapping(data, "config")
        mode = _as_str(
            _mapping(raw.get("setup"), "setup").get("mode"), "recommended"
        )
        # Missing sections inherit the preset the file claims to use.  This
        # keeps old or hand-edited files forward-compatible.
        merged = _deep_merge(base_settings(mode).to_dict(), raw)
        return cls(
            version=_as_int(
                merged.get("version"), CONFIG_VERSION, minimum=1
            ),
            setup=SetupSettings.from_dict(merged.get("setup")),
            agent=AgentSettings.from_dict(merged.get("agent")),
            provider=ProviderSettings.from_dict(merged.get("provider")),
            channel=ChannelSettings.from_dict(merged.get("channel")),
            features=FeatureSettings.from_dict(merged.get("features")),
            limits=LimitSettings.from_dict(merged.get("limits")),
            paths=PathSettings.from_dict(merged.get("paths")),
        )


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def base_settings(mode: str) -> Settings:
    """Return the built-in preset for ``recommended`` or ``advanced``."""
    if mode == "advanced":
        return advanced_settings()
    if mode == "legacy":
        return legacy_settings()
    return recommended_settings()


def recommended_settings() -> Settings:
    """Everything on, tuned for daily use."""
    return Settings(
        setup=SetupSettings(mode="recommended"),
        agent=AgentSettings(
            name="Sora",
            icon="☁️",
            personality=PersonalitySettings(
                base="friendly", warmth="very_warm"
            ),
        ),
        provider=ProviderSettings(
            kind="openai",
            model="gpt-5.4-mini",
            temperature=0.7,
            context_window=272_000,
        ),
        channel=ChannelSettings(),
        features=FeatureSettings(
            memory=MemorySettings(
                enabled=True,
                sidecar=True,
                auto_download=True,
                dream=True,
                backup=True,
                wake=True,
            ),
            identity=True,
            clock=True,
            calendar=True,
            sessions=True,
            tools="demo",
            streaming=True,
            approval="ask",
            quality=True,
        ),
        limits=LimitSettings(
            max_steps=12,
            max_turns=80,
            provider_timeout=60.0,
            provider_retries=3,
            tool_timeout=60.0,
            approval_timeout=300.0,
        ),
    )


def advanced_settings() -> Settings:
    """Every optional subsystem off; policy gates off too."""
    return Settings(
        setup=SetupSettings(mode="advanced"),
        agent=AgentSettings(
            name="Sora",
            icon="☁️",
            personality=PersonalitySettings(
                base="neutral", warmth="neutral"
            ),
        ),
        provider=ProviderSettings(
            kind="openai",
            model="gpt-5.4-mini",
            temperature=0.1,
            context_window=272_000,
        ),
        channel=ChannelSettings(),
        features=FeatureSettings(
            memory=MemorySettings(enabled=False),
            identity=False,
            clock=False,
            calendar=False,
            sessions=False,
            tools="none",
            streaming=False,
            approval="off",
            quality=False,
        ),
        limits=LimitSettings(
            max_steps=8,
            max_turns=40,
            provider_timeout=30.0,
            provider_retries=2,
            tool_timeout=30.0,
            approval_timeout=300.0,
        ),
    )


def legacy_settings() -> Settings:
    """The defaults ``sora-gateway`` used before config files existed.

    Used only when no config file is present, so an environment-only install
    keeps its exact previous behaviour.
    """
    return Settings(
        setup=SetupSettings(mode="legacy"),
        agent=AgentSettings(
            name="Sora",
            icon="☁️",
            personality=PersonalitySettings(
                base="neutral", warmth="neutral"
            ),
        ),
        provider=ProviderSettings(
            kind="mock",
            model="gpt-4o-mini",
            temperature=0.7,
            context_window=128_000,
        ),
        channel=ChannelSettings(),
        features=FeatureSettings(
            memory=MemorySettings(enabled=False),
            identity=False,
            clock=True,
            calendar=True,
            sessions=True,
            tools="none",
            streaming=True,
            approval="ask",
            quality=True,
        ),
        limits=LimitSettings(
            max_steps=8,
            max_turns=40,
            provider_timeout=30.0,
            provider_retries=2,
            tool_timeout=30.0,
            approval_timeout=300.0,
        ),
    )


# ---------------------------------------------------------------------------
# Loading and saving
# ---------------------------------------------------------------------------


def config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    configured = os.getenv(CONFIG_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CONFIG_PATH


def secrets_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    configured = os.getenv(SECRETS_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_SECRETS_PATH


def load_settings(path: str | Path | None = None) -> Settings | None:
    """Load settings, or return ``None`` when the file does not exist."""
    target = config_path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SettingsError(f"cannot read {target}: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise SettingsError(f"{target} is not valid JSON: {exc}") from exc
    return Settings.from_dict(data)


def save_settings(
    settings: Settings, path: str | Path | None = None
) -> Path:
    target = config_path(path)
    # Round-trip through the parser so mutated dataclasses are validated and
    # normalized before anything touches disk.
    normalized = Settings.from_dict(settings.to_dict())
    _atomic_write(
        target,
        json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2) + "\n",
        0o600,
    )
    return target


def load_secrets(path: str | Path | None = None) -> dict[str, str]:
    """Parse a small ``KEY=VALUE`` env file.  Missing files yield ``{}``."""
    target = secrets_path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsError(f"cannot read {target}: {exc}") from exc
    values: dict[str, str] = {}
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[len("export "):].strip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote(value.strip())
    return values


def save_secrets(
    values: Mapping[str, str], path: str | Path | None = None
) -> Path:
    target = secrets_path(path)
    lines = [
        "# SoraAgent secrets. Generated by `sora setup`.",
        "# This file is private; do not commit or share it.",
    ]
    for key in sorted(values):
        value = str(values[key])
        if not value:
            continue
        if any(char in value for char in ("\n", "\r")):
            raise SettingsError(f"secret {key} must be a single line")
        lines.append(f"{key}={_quote(value)}")
    _atomic_write(target, "\n".join(lines) + "\n", 0o600)
    return target


def update_secrets(
    updates: Mapping[str, str], path: str | Path | None = None
) -> dict[str, str]:
    """Merge ``updates`` into the secrets file and return the result.

    A value of ``""`` removes the key.
    """
    merged = load_secrets(path)
    for key, value in updates.items():
        if value:
            merged[key] = str(value)
        else:
            merged.pop(key, None)
    save_secrets(merged, path)
    return merged


def _quote(value: str) -> str:
    if not value or any(
        char.isspace() or char in {'"', "'", "#", "\\"} for char in value
    ):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    # Drop an unquoted trailing comment only when it is clearly separated.
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _atomic_write(target: Path, text: str, mode: int) -> None:
    parent_existed = target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Only harden the default private directory (or a directory we just made).
    # An explicit SORA_CONFIG may point into a shared project directory whose
    # permissions we must not change.
    if not parent_existed or target.parent == DEFAULT_ROOT:
        try:
            os.chmod(target.parent, 0o700)
        except OSError:
            pass
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, mode)
        except OSError:
            pass
        os.replace(temporary_path, target)
    except OSError as exc:
        raise SettingsError(f"cannot write {target}: {exc}") from exc
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


_T = TypeVar("_T")


def resolved(*values: _T | None) -> _T | None:
    """Return the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def env_flag(name: str) -> bool | None:
    """Read an optional boolean environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None
