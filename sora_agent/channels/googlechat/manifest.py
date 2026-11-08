"""Google Chat channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="googlechat",
    display_name="Google Chat",
    runtime=f"{__package__}.runtime:GoogleChatChannel",
)
