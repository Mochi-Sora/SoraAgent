"""Focused tests for the compact tool-schema generator."""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Literal

from sora_agent.tools import Tool


class Color(Enum):
    RED = "red"
    BLUE = "blue"


def test_scalar_defaults_are_emitted() -> None:
    def example(required: str, limit: int = 5, note: str | None = None) -> str:
        return required

    schema = Tool(example, "example", "Example tool.").schema
    properties = schema["parameters"]["properties"]

    assert properties["required"] == {"type": "string"}
    assert properties["limit"] == {"type": "integer", "default": 5}
    assert properties["note"] == {"type": "string"}
    assert schema["parameters"]["required"] == ["required"]


def test_literal_becomes_enum() -> None:
    def example(mode: Literal["auto", "daily"] = "auto") -> str:
        return mode

    schema = Tool(example, "example", "Example tool.").schema

    assert schema["parameters"]["properties"]["mode"] == {
        "type": "string",
        "enum": ["auto", "daily"],
        "default": "auto",
    }


def test_enum_class_becomes_enum() -> None:
    def example(color: Color) -> str:
        return color.value

    schema = Tool(example, "example", "Example tool.").schema

    assert schema["parameters"]["properties"]["color"] == {
        "type": "string",
        "enum": ["red", "blue"],
    }


def test_annotated_adds_only_a_description() -> None:
    def example(
        order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
        note: str,
    ) -> str:
        return order_id + note

    schema = Tool(example, "example", "Example tool.").schema
    properties = schema["parameters"]["properties"]

    assert properties["order_id"] == {
        "type": "string",
        "description": "Purchase order id, e.g. PO-1234",
    }
    assert properties["note"] == {"type": "string"}


def test_optional_annotated_keeps_the_description() -> None:
    def example(
        value: Annotated[str, "External identifier"] | None,
    ) -> str:
        return value or ""

    schema = Tool(example, "example", "Example tool.").schema

    assert schema["parameters"]["properties"]["value"] == {
        "type": "string",
        "description": "External identifier",
    }


def test_variadic_parameters_are_ignored() -> None:
    def example(order_id: str, *args: str, **kwargs: Any) -> str:
        return order_id

    schema = Tool(example, "example", "Example tool.").schema

    assert schema["parameters"]["properties"] == {
        "order_id": {"type": "string"}
    }
    assert schema["parameters"]["required"] == ["order_id"]


def test_non_scalar_defaults_are_skipped() -> None:
    def example(values: tuple[str, ...] = ("a",)) -> str:
        return "".join(values)

    schema = Tool(example, "example", "Example tool.").schema

    assert schema["parameters"]["properties"]["values"] == {"type": "array"}


def test_approval_marker_is_added_once() -> None:
    guarded = Tool(
        lambda: "ok", "guarded", "Do a thing.", needs_approval=True
    ).schema
    documented = Tool(
        lambda: "ok", "documented", "Needs approval.", needs_approval=True
    ).schema

    assert guarded["description"] == "[approval required] Do a thing."
    assert documented["description"] == "Needs approval."


def test_schema_is_json_serializable() -> None:
    def example(mode: Literal["a", "b"] = "a") -> str:
        return mode

    json.dumps(Tool(example, "example", "Example tool.").schema)
