"""Dependency-free descriptors for channel packages.

Discovery imports a package's ``manifest.py`` before it knows whether the
platform dependency is installed, so a manifest must never import its runtime
module or a platform SDK.  The runtime is an import string resolved on demand.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseChannel

_CHANNEL_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_TARGET_RE = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")


@dataclass(frozen=True)
class ChannelPlugin:
    """Metadata for one channel package."""

    name: str
    display_name: str
    runtime: str
    dependencies: tuple[str, ...] = ()
    default_enabled: bool = False
    settings_visible: bool = True

    def __post_init__(self) -> None:
        if _CHANNEL_NAME_RE.fullmatch(self.name) is None:
            raise ValueError(
                "channel name must start with a letter and contain only "
                "letters, digits, or underscores"
            )
        if _TARGET_RE.fullmatch(self.runtime) is None:
            raise ValueError(
                f"channel {self.name!r} runtime must be 'module:Class'"
            )
        if not isinstance(self.dependencies, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.dependencies
        ):
            raise TypeError("channel dependencies must be a tuple of requirements")

    def load_channel_class(self) -> type[BaseChannel]:
        """Resolve, validate, and return the runtime class."""
        from .base import BaseChannel

        module_name, _, attribute = self.runtime.partition(":")
        module = importlib.import_module(module_name)
        channel_cls: object = getattr(module, attribute, None)
        if (
            not isinstance(channel_cls, type)
            or not issubclass(channel_cls, BaseChannel)
            or channel_cls is BaseChannel
        ):
            raise ImportError(
                f"channel {self.name!r} runtime {self.runtime!r} is not a "
                "BaseChannel subclass"
            )
        if channel_cls.name != self.name:
            raise ImportError(
                f"channel {self.name!r} runtime declares name "
                f"{channel_cls.name!r}"
            )
        return channel_cls
