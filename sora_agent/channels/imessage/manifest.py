"""iMessage channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="imessage",
    display_name="iMessage",
    runtime=f"{__package__}.runtime:IMessageChannel",
)
