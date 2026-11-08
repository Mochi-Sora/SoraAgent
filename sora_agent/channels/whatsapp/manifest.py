"""WhatsApp channel descriptor (no SDK imports)."""

from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="whatsapp",
    display_name="WhatsApp Cloud API",
    runtime=f"{__package__}.runtime:WhatsAppChannel",
)
