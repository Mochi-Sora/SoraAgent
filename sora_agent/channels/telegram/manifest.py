"""Telegram channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="telegram",
    display_name="Telegram",
    runtime=f"{__package__}.runtime:TelegramChannel",
)
