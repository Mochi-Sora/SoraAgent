"""A tiny tool registry and the policy-aware executor.

Decorate a function with `@tool` and it becomes callable by the agent. Set
`needs_approval=True` to route that action through the human-approval gate.

`ToolExecutor` is the only place that turns a model tool call into a result:
validation, approval, execution, truncation, and error observations all happen
here.  The runner stays a plain model/tool loop; quality policy is a pre-tool
guard on the runner, so a rejected turn never reaches this executor.
"""

from __future__ import annotations

import inspect
import json
import logging
import types
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .isolation import IsolationError, ToolTimeout, run_isolated
from .providers import ToolCall

log = logging.getLogger(__name__)

_UNION_ORIGINS = (Union,) + (
    (types.UnionType,) if hasattr(types, "UnionType") else ()
)
_APPROVAL_MARKER = "[approval required] "
_JSON_SCALARS = (str, int, float, bool)


class Tool:
    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        needs_approval: bool = False,
        isolated: bool = False,
        requires: tuple[str, ...] = (),
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.needs_approval = needs_approval
        self.isolated = isolated
        self.requires = tuple(str(item) for item in requires)

    @property
    def schema(self) -> dict[str, Any]:
        params = inspect.signature(self.fn).parameters
        try:
            hints = get_type_hints(self.fn, include_extras=True)
        except Exception:  # noqa: BLE001 - unresolved annotations fall back to strings
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in params.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            prop = _json_schema(hints.get(name, param.annotation))
            if (
                param.default is not inspect.Parameter.empty
                and param.default is not None
                and isinstance(param.default, _JSON_SCALARS)
            ):
                prop["default"] = param.default
            properties[name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(name)

        description = self.description
        if self.needs_approval and "approval" not in description.casefold():
            description = _APPROVAL_MARKER + description

        return {
            "name": self.name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def call(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def _json_schema(annotation: Any) -> dict[str, Any]:
    """Return a compact JSON Schema fragment for one annotation."""
    extras: dict[str, Any] = {}
    metadata = getattr(annotation, "__metadata__", ())
    if metadata:
        args = get_args(annotation)
        if args:
            annotation = args[0]
        description = next(
            (item for item in metadata if isinstance(item, str)), ""
        )
        if description:
            extras["description"] = description

    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        schema = _json_schema(args[0]) if args else {"type": "string"}
        schema.update(extras)
        return schema

    if origin is Literal:
        values = [
            value for value in get_args(annotation) if value is not None
        ]
        schema = _enum_schema(values)
        if not schema:
            schema = {"type": "string"}
        schema.update(extras)
        return schema

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        schema = _enum_schema([member.value for member in annotation])
        if not schema:
            schema = {"type": "string"}
        schema.update(extras)
        return schema

    if origin in (list, tuple, set):
        schema = {"type": "array"}
    elif origin is dict:
        schema = {"type": "object"}
    else:
        schema = {
            "type": {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                dict: "object",
                list: "array",
            }.get(annotation, "string")
        }
    schema.update(extras)
    return schema


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    if not values or not all(
        isinstance(value, _JSON_SCALARS) for value in values
    ):
        return {}
    schema: dict[str, Any] = {"enum": values}
    type_name = _scalar_type(values)
    if type_name:
        schema["type"] = type_name
    return schema


def _scalar_type(values: list[Any]) -> str | None:
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in values
    ):
        return "integer"
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return "number"
    if all(isinstance(value, str) for value in values):
        return "string"
    return None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def copy(self) -> ToolRegistry:
        """Return a shallow clone so one run's discovery cannot leak."""
        clone = ToolRegistry()
        clone._tools = dict(self._tools)
        return clone


@dataclass
class ToolResult:
    """The normalized outcome of one tool call."""

    call: ToolCall
    content: str
    is_error: bool = False
    approval_required: bool = False
    approved: bool = False
    executed: bool = False
    stopped: bool = False
    stop_reason: str = ""
    report: Any = None
    skipped: bool = False
    isolated: bool = False


class ToolExecutor:
    """Validate, gate, execute, and normalize a model tool call."""

    def __init__(
        self,
        registry: ToolRegistry,
        approval: Any = None,
        max_result_chars: int = 4000,
        after_execute: Callable[[ToolCall, ToolResult], None] | None = None,
        tool_timeout: float = 30.0,
    ) -> None:
        self.registry = registry
        self.approval = approval
        self.max_result_chars = max(0, int(max_result_chars))
        self.after_execute = after_execute
        self.tool_timeout = max(0.1, float(tool_timeout))

    def execute(self, call: ToolCall) -> ToolResult:
        result = ToolResult(call=call, content="")

        try:
            tool = self.registry.get(call.name)
        except KeyError:
            result.content = f"Unknown tool: {call.name}"
            result.is_error = True
            return result

        if call.error:
            result.content = f"Invalid arguments for {call.name}: {call.error}"
            result.is_error = True
            return result

        if not isinstance(call.arguments, dict):
            result.content = f"Invalid arguments for {call.name}: expected an object"
            result.is_error = True
            return result

        try:
            _validate_arguments(tool, call.arguments)
        except TypeError as exc:
            result.content = f"Invalid arguments for {call.name}: {exc}"
            result.is_error = True
            return result

        if getattr(tool, "needs_approval", False):
            result.approval_required = True
            if self.approval is None:
                result.content = "approval required but no approval gate is configured"
                result.is_error = True
                result.stopped = True
                result.stop_reason = "approval_rejected"
                return result
            try:
                approved = bool(
                    self.approval.request(call.name, _render_call(call))
                )
            except Exception as exc:  # noqa: BLE001 - approval must fail closed
                result.content = f"approval gate failed: {exc}"
                result.is_error = True
                result.stopped = True
                result.stop_reason = "approval_rejected"
                return result
            result.approved = approved
            if not approved:
                result.content = "rejected by human approval gate"
                result.stopped = True
                result.stop_reason = "approval_rejected"
                return result

        result.isolated = bool(getattr(tool, "isolated", False))
        try:
            if result.isolated:
                raw = run_isolated(
                    tool.fn, call.arguments, self.tool_timeout
                )
            else:
                raw = tool.call(**call.arguments)
        except ToolTimeout as exc:
            result.content = str(exc)
            result.is_error = True
            return result
        except IsolationError as exc:
            result.content = f"isolated tool failed: {exc}"
            result.is_error = True
            return result
        except Exception as exc:  # noqa: BLE001 - tool failures become observations
            result.content = f"tool call failed: {exc}"
            result.is_error = True
            return result

        result.executed = True
        result.content = _truncate(_as_text(raw), self.max_result_chars)

        if self.after_execute is not None:
            try:
                self.after_execute(call, result)
            except Exception as exc:  # noqa: BLE001 - policy hooks must not break the loop
                result.content += f" | post-tool hook: {exc}"

        if self.approval is not None:
            try:
                self.approval.log(call.name, "executed", result.content)
            except Exception as exc:  # noqa: BLE001 - audit is best-effort
                log.debug("approval audit log failed: %s", exc)
        return result


def _validate_arguments(tool: Any, arguments: dict[str, Any]) -> None:
    target = getattr(tool, "fn", None) or getattr(tool, "call", None)
    if target is None:
        return
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return
    signature.bind(**arguments)


def _render_call(call: ToolCall) -> str:
    args = ", ".join(
        f"{key}={json.dumps(value, default=str)}"
        for key, value in call.arguments.items()
    )
    return f"ACTION: {call.name}({args})"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


registry = ToolRegistry()


def tool(
    name: str, description: str, needs_approval: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.add(Tool(fn, name, description, needs_approval))
        return fn

    return deco


# --- example tools -----------------------------------------------------------


@tool("lookup_order", "Return a purchase order's status and details.")
def lookup_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
) -> str:
    return f"PO {order_id}: status=open, supplier=ACME, ship_date=2026-09-01"


@tool("expedite_order", "Expedite a purchase order.", needs_approval=True)
def expedite_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
) -> str:
    return f"PO {order_id}: marked expedited + rescheduled ship_date to 2026-08-25"


@tool("cancel_order", "Cancel a purchase order.", needs_approval=True)
def cancel_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
) -> str:
    return f"PO {order_id}: cancelled"
