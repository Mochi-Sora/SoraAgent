"""Discover channel packages under ``sora_agent.channels``."""

from __future__ import annotations

import importlib
import logging
import pkgutil

from .manifest import ChannelPlugin

log = logging.getLogger(__name__)

_PACKAGE = "sora_agent.channels"


def channel_package_names() -> list[str]:
    """Return the subpackage names that may contain a channel."""
    package = importlib.import_module(_PACKAGE)
    return sorted(
        name
        for _, name, is_package in pkgutil.iter_modules(package.__path__)
        if is_package
    )


def load_plugin(name: str) -> ChannelPlugin:
    """Load one channel descriptor without importing its runtime."""
    module = importlib.import_module(f"{_PACKAGE}.{name}.manifest")
    plugin = getattr(module, "PLUGIN", None)
    if not isinstance(plugin, ChannelPlugin):
        # Discovery and the CLI catch ImportError for an unusable package.
        raise ImportError(  # noqa: TRY004
            f"channel package {name!r} has no PLUGIN descriptor"
        )
    if plugin.name != name:
        raise ImportError(
            f"channel package {name!r} declares name {plugin.name!r}"
        )
    return plugin


def discover_plugins(
    enabled_names: set[str] | None = None,
) -> dict[str, ChannelPlugin]:
    """Load descriptors for every channel package, skipping broken ones."""
    plugins: dict[str, ChannelPlugin] = {}
    for name in channel_package_names():
        if enabled_names is not None and name not in enabled_names:
            continue
        try:
            plugins[name] = load_plugin(name)
        except Exception as exc:  # noqa: BLE001 - one bad package must not hide the rest
            log.warning("failed to load channel %r: %s", name, exc)
    return plugins


def load_channel_class(name: str):
    """Resolve one channel's runtime class."""
    return load_plugin(name).load_channel_class()
