"""Slack channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="slack",
    display_name="Slack",
    runtime=f"{__package__}.runtime:SlackChannel",
)
