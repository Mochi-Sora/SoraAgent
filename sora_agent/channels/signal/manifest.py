"""Signal channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="signal",
    display_name="Signal",
    runtime=f"{__package__}.runtime:SignalChannel",
)
