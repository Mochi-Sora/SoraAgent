"""Discord channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="discord",
    display_name="Discord",
    runtime=f"{__package__}.runtime:DiscordChannel",
)
