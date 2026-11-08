"""Matrix channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="matrix",
    display_name="Matrix",
    runtime=f"{__package__}.runtime:MatrixChannel",
)
