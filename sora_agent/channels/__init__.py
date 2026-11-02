"""Chat channels: transports that feed the agent and deliver its replies."""

from .approval import (
    ApprovalBroker,
    ChannelApprovalGate,
    PendingApproval,
    parse_decision,
)
from .base import BaseChannel
from .bus import MessageBus
from .chat import ChatHandler
from .manager import ChannelManager
from .manifest import ChannelPlugin
from .registry import (
    channel_package_names,
    discover_plugins,
    load_channel_class,
    load_plugin,
)
from .types import InboundMessage, OutboundMessage

__all__ = [
    "ApprovalBroker",
    "BaseChannel",
    "ChannelApprovalGate",
    "ChannelManager",
    "ChannelPlugin",
    "ChatHandler",
    "InboundMessage",
    "MessageBus",
    "OutboundMessage",
    "PendingApproval",
    "channel_package_names",
    "discover_plugins",
    "load_channel_class",
    "load_plugin",
    "parse_decision",
]
