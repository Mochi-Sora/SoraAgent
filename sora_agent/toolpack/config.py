"""Configuration for the built-in Hermes-style core tool pack."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


WEB_SEARCH_PROVIDERS = ("auto", "duckduckgo", "tavily", "brave", "custom", "off")


@dataclass
class WebSearchConfig:
    """Resolved web-search provider and credentials.

    Credentials are deliberately excluded from ``repr`` so a debug log cannot
    accidentally print an API key.
    """

    provider: str = "auto"
    url: str = ""
    tavily_api_key: str = field(default="", repr=False)
    brave_api_key: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)

    def resolved_provider(self) -> str:
        provider = (self.provider or "auto").strip().lower()
        if provider == "auto":
            if self.url.strip():
                return "custom"
            if self.tavily_api_key.strip():
                return "tavily"
            if self.brave_api_key.strip():
                return "brave"
            return "duckduckgo"
        return provider


@dataclass
class ToolPackConfig:
    """Limits and roots for the core tool pack.

    ``root`` is the workspace the file and shell tools operate in.  Keeping
    writes inside that root is the default; terminal commands still inherit
    the host environment, so they remain approval-gated.
    """

    root: Path = field(default_factory=Path.cwd)
    restrict_to_root: bool = True
    roots: tuple[Path, ...] = ()
    allow_tmp: bool = False
    max_read_bytes: int = 2_000_000
    max_read_chars: int = 40_000
    max_write_bytes: int = 2_000_000
    max_search_results: int = 100
    max_search_file_bytes: int = 1_000_000
    search_timeout: float = 30.0
    max_terminal_output: int = 20_000
    terminal_timeout: float = 120.0
    code_timeout: float = 60.0
    web_timeout: float = 20.0
    web_max_chars: int = 20_000
    allow_private_network: bool = False
    search_results: int = 5
    workspace_env: str = ""
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        extra: list[Path] = []
        for item in self.roots:
            text = str(item).strip()
            if text:
                extra.append(Path(text).expanduser().resolve())
        if self.allow_tmp:
            extra.append(Path(tempfile.gettempdir()).expanduser().resolve())
        self.roots = tuple(dict.fromkeys(extra))
        self.max_read_bytes = max(1, int(self.max_read_bytes))
        self.max_read_chars = max(200, int(self.max_read_chars))
        self.max_write_bytes = max(1, int(self.max_write_bytes))
        self.max_search_results = max(1, int(self.max_search_results))
        self.max_search_file_bytes = max(1, int(self.max_search_file_bytes))
        self.search_timeout = max(0.1, float(self.search_timeout))
        self.max_terminal_output = max(200, int(self.max_terminal_output))
        self.terminal_timeout = max(0.1, float(self.terminal_timeout))
        self.code_timeout = max(0.1, float(self.code_timeout))
        self.web_timeout = max(0.1, float(self.web_timeout))
        self.web_max_chars = max(500, int(self.web_max_chars))
        self.search_results = max(1, min(100, int(self.search_results)))

    def allowed_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys((self.root, *self.roots)))

    def resolve(self, value: str, *, must_exist: bool = False) -> Path:
        """Resolve a path relative to the workspace and enforce the roots."""

        from .safety import resolve_path

        return resolve_path(self, value, must_exist=must_exist)

    @classmethod
    def from_env(cls, root: Path | None = None) -> ToolPackConfig:
        configured_root = (
            root
            or os.getenv("SORA_TOOLPACK_ROOT")
            or os.getenv("SORA_WORKSPACE")
            or Path.cwd()
        )
        raw_roots = os.getenv("SORA_TOOLPACK_ROOTS", "").strip()
        extra_roots = tuple(
            Path(part.strip())
            for part in raw_roots.split(os.pathsep)
            if part.strip()
        )
        web = WebSearchConfig(
            provider=os.getenv("SORA_WEB_SEARCH_PROVIDER", "auto").strip()
            or "auto",
            url=os.getenv("SORA_WEB_SEARCH_URL", "").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            brave_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
            api_key=os.getenv("SORA_WEB_SEARCH_API_KEY", "").strip(),
        )
        return cls(
            root=Path(configured_root),
            restrict_to_root=_env_bool("SORA_TOOLPACK_RESTRICT", True),
            roots=extra_roots,
            allow_tmp=_env_bool("SORA_TOOLPACK_ALLOW_TMP", False),
            max_read_bytes=_env_int("SORA_TOOLPACK_MAX_READ_BYTES", 2_000_000),
            max_read_chars=_env_int(
                "SORA_TOOLPACK_MAX_READ_CHARS", 40_000, minimum=200
            ),
            max_write_bytes=_env_int("SORA_TOOLPACK_MAX_WRITE_BYTES", 2_000_000),
            max_search_results=_env_int("SORA_TOOLPACK_MAX_SEARCH_RESULTS", 100),
            max_search_file_bytes=_env_int(
                "SORA_TOOLPACK_MAX_SEARCH_FILE_BYTES", 1_000_000
            ),
            search_timeout=_env_float("SORA_TOOLPACK_SEARCH_TIMEOUT", 30.0),
            max_terminal_output=_env_int(
                "SORA_TOOLPACK_MAX_TERMINAL_OUTPUT", 20_000, minimum=200
            ),
            terminal_timeout=_env_float("SORA_TOOLPACK_TERMINAL_TIMEOUT", 120.0),
            code_timeout=_env_float("SORA_TOOLPACK_CODE_TIMEOUT", 60.0),
            web_timeout=_env_float("SORA_TOOLPACK_WEB_TIMEOUT", 20.0),
            web_max_chars=_env_int("SORA_TOOLPACK_WEB_MAX_CHARS", 20_000, minimum=500),
            allow_private_network=_env_bool(
                "SORA_TOOLPACK_ALLOW_PRIVATE_NETWORK", False
            ),
            search_results=_env_int("SORA_TOOLPACK_SEARCH_RESULTS", 5),
            web_search=web,
        )
