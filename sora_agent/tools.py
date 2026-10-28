"""A tiny tool registry.

Decorate a function with `@tool` and it becomes callable by the agent. Set
`needs_approval=True` to route that action through the human-approval gate.
"""

from __future__ import annotations

import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


_UNION_ORIGINS = (Union,) + (
    (types.UnionType,) if hasattr(types, "UnionType") else ()
)


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
    def schema(self) -> dict[str, Any]:
        params = inspect.signature(self.fn).parameters
        try:
            hints = get_type_hints(self.fn)
        except Exception:  # noqa: BLE001 - unresolved annotations fall back to strings
            hints = {}
        props = {
            name: {"type": _json_type(hints.get(name, param.annotation))}
            for name, param in params.items()
        }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": [
                    name
                    for name, param in params.items()
                    if param.default is inspect.Parameter.empty
                ],
            },
        }

    def call(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def _json_type(annotation: Any) -> str:
    if annotation in (inspect.Parameter.empty, Any):
        return "string"
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _json_type(args[0]) if args else "string"
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }.get(annotation, "string")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def all(self) -> list[Tool]:
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
