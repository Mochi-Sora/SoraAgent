"""A tiny tool registry.

Decorate a function with `@tool` and it becomes callable by the agent. Set
`needs_approval=True` to route that action through the human-approval gate.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List


class Tool:
    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        needs_approval: bool = False,
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.needs_approval = needs_approval

    @property
    def schema(self) -> Dict[str, Any]:
        params = inspect.signature(self.fn).parameters
        props = {
            n: {"type": p.annotation.__name__ if p.annotation is not inspect.Parameter.empty else "string"}
            for n, p in params.items()
        }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(params.keys()),
            },
        }

    def call(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def all(self) -> List[Tool]:
        return list(self._tools.values())


registry = ToolRegistry()


def tool(name: str, description: str, needs_approval: bool = False):
    def deco(fn: Callable) -> Callable:
        registry.add(Tool(fn, name, description, needs_approval))
        return fn

    return deco


# --- example tools -----------------------------------------------------------


@tool("lookup_order", "Return status and details for a purchase order id.")
def lookup_order(order_id: str) -> str:
    return f"PO {order_id}: status=open, supplier=ACME, ship_date=2026-09-01"


@tool("expedite_order", "Mark an order as expedited (needs human approval).", needs_approval=True)
def expedite_order(order_id: str) -> str:
    return f"PO {order_id}: marked expedited + rescheduled ship_date to 2026-08-25"


@tool("cancel_order", "Cancel a purchase order (needs human approval).", needs_approval=True)
def cancel_order(order_id: str) -> str:
    return f"PO {order_id}: cancelled"
