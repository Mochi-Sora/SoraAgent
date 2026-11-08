"""Feishu / Lark channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="feishu",
    display_name="Feishu / Lark",
    runtime=f"{__package__}.runtime:FeishuChannel",
)
