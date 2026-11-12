"""Sample purchase-order tools used by the examples.

These are fixtures, not a real order system. SoraAgent's built-in core tool
pack covers file, shell, web, planning, code-execution, and delegation work;
use this module as the model for registering your own domain tools on a
``ToolRegistry`` (or with the ``@tool`` decorator).
"""

from __future__ import annotations

from typing import Annotated

from sora_agent.tools import Tool, ToolRegistry


def lookup_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
) -> str:
    return f"PO {order_id}: status=open, supplier=ACME, ship_date=2026-09-01"


def expedite_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
) -> str:
    return f"PO {order_id}: marked expedited + rescheduled ship_date to 2026-08-25"


def cancel_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
) -> str:
    return f"PO {order_id}: cancelled"


registry = ToolRegistry()
registry.add(
    Tool(
        lookup_order,
        "lookup_order",
        "Return a purchase order's status and details.",
        aliases=("status", "check order", "order details"),
    )
)
registry.add(
    Tool(
        expedite_order,
        "expedite_order",
        "Expedite a purchase order.",
        needs_approval=True,
        aliases=("rush", "speed up", "late order"),
    )
)
registry.add(
    Tool(
        cancel_order,
        "cancel_order",
        "Cancel a purchase order.",
        needs_approval=True,
        aliases=("void", "remove order"),
    )
)
