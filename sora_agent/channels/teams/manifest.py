"""Microsoft Teams channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="teams",
    display_name="Microsoft Teams",
    runtime=f"{__package__}.runtime:TeamsChannel",
)
