"""Model provider abstraction.

The agent loop never talks to a vendor directly. Swap the provider and everything else works.
Use `MockProvider` to run demos/CIs offline.

Providers may implement ``complete_turn`` for native tool calls.  ``complete`` remains the
text-only view of the same operation; text ``ACTION:`` lines are parsed here, at the provider
edge, so the runner only ever sees structured turns.
"""

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Union

if TYPE_CHECKING:
    from .tools import Tool


_ACTION_RE = re.compile(
    r"ACTION:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", re.DOTALL
)
_ORDER_RE = re.compile(r"\b(PO-\d+)\b", re.IGNORECASE)


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    error: str = ""


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""


@dataclass
class ModelTurn:
    """One assistant turn: visible text plus any structured tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class TurnProvider(Protocol):
    def complete_turn(
        self, messages: list[Message], tools: list[Tool]
    ) -> ModelTurn:
        """Return the assistant's text and, optionally, native tool calls."""
        ...


class TextProvider(Protocol):
    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        """Return only the assistant's text response."""
        ...


# The runtime accepts either native turn providers or legacy text providers.
ModelProvider = Union[TurnProvider, TextProvider]


# ---------------------------------------------------------------------------
# Provider-edge adapters
# ---------------------------------------------------------------------------


def provider_turn(
    provider: Any,
    messages: Sequence[Message],
    tools: Sequence[Any],
) -> ModelTurn:
    """Ask any provider for a structured turn.

    Providers with ``complete_turn`` are used directly.  Older providers that only
    implement ``complete`` get their text parsed for a single legacy ``ACTION:`` line.
    This is the only place where the text protocol is understood.
    """

    method = getattr(provider, "complete_turn", None)
    if callable(method):
        return _coerce_turn(method(list(messages), list(tools)))

    complete = getattr(provider, "complete", None)
    if not callable(complete):
        raise TypeError(
            f"provider {type(provider).__name__} has neither complete_turn() "
            "nor complete()"
        )
    return turn_from_text(complete(list(messages), list(tools)))


def provider_text(
    provider: Any,
    messages: Sequence[Message],
    tools: Sequence[Any],
) -> str:
    """Return only the text of a provider call, for repair loops."""

    method = getattr(provider, "complete", None)
    if callable(method):
        return _as_text(method(list(messages), list(tools)))
    return provider_turn(provider, messages, tools).text


def turn_from_text(raw: Any) -> ModelTurn:
    text = _as_text(raw)
    return ModelTurn(text=text, tool_calls=parse_action_calls(text))


def parse_action_calls(text: str) -> list[ToolCall]:
    """Parse the legacy single-action text protocol.

    The text format deliberately supports one action per turn, matching the old
    loop.  Native providers should emit structured tool calls instead.
    """

    if not text:
        return []
    match = _ACTION_RE.search(text)
    if not match:
        return []
    raw = match.group(2)
    return [
        ToolCall(
            id=f"text_{uuid.uuid4().hex[:12]}",
            name=match.group(1),
            arguments=parse_kwargs(raw),
            raw_arguments=raw,
        )
    ]


def parse_kwargs(raw: str) -> dict[str, Any]:
    """Parse ``k="v", n=3`` into a dict (quote-aware top-level comma split)."""

    kwargs: dict[str, Any] = {}
    if not raw.strip():
        return kwargs
    for part in _split_top(raw, ","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        value = value.strip()
        try:
            kwargs[key.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key.strip()] = value.strip("\"'")
    return kwargs


def _split_top(text: str, separator: str) -> list[str]:
    out: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote = ""
    for char in text:
        if char == quote:
            quote = ""
            buffer.append(char)
            continue
        if not quote:
            if char == separator and depth == 0:
                out.append("".join(buffer))
                buffer = []
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char in "\"'":
                quote = char
        buffer.append(char)
    if buffer:
        out.append("".join(buffer))
    return out


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, ModelTurn):
        return value.text
    return str(value)


def _coerce_turn(value: Any) -> ModelTurn:
    if isinstance(value, ModelTurn):
        return value
    if isinstance(value, str):
        return turn_from_text(value)
    if value is None:
        return ModelTurn()
    return ModelTurn(
        text=_as_text(getattr(value, "text", "")),
        tool_calls=list(getattr(value, "tool_calls", []) or []),
    )


# ---------------------------------------------------------------------------
# Offline provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Returns canned turns so the demo runs without an API key.

    It follows the same contract as a real model: recall memory first when the
    tool is visible, discover a hidden business tool with ``find_tools``, then
    call the tool and answer.  That keeps the offline demo on the production
    path instead of bypassing the discovery gates.
    """

    def complete_turn(
        self, messages: list[Message], tools: list[Tool]
    ) -> ModelTurn:
        visible = {
            str(getattr(tool, "name", "")) for tool in tools
        }
        tool_names = [
            message.name for message in messages if message.role == "tool"
        ]
        last_user = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        low = last_user.lower()
        order_id = _extract_order_id(last_user)
        expedite = (
            "expedite" in low
            or "stuck" in low
            or "can't ship" in low
            or "cannot ship" in low
        )
        target = "expedite_order" if expedite else "lookup_order"
        need = (
            "expedite a delayed purchase order"
            if expedite
            else "look up purchase order status and details"
        )

        if "memory_recall" in visible and not tool_names:
            return ModelTurn(
                text="Let me recall relevant context before acting.",
                tool_calls=[
                    ToolCall(
                        id="call_recall",
                        name="memory_recall",
                        arguments={
                            "query": last_user or "task context",
                            "scope": "auto",
                            "k": 8,
                        },
                    )
                ],
            )

        if target in tool_names:
            if expedite:
                text = (
                    f"{order_id} was expedited after approval. "
                    "The requested tool call completed successfully."
                )
            else:
                text = (
                    f"{order_id} was checked. "
                    "The requested tool call completed successfully."
                )
            return ModelTurn(text=text)

        if target not in visible:
            if "find_tools" in visible and "find_tools" not in tool_names:
                return ModelTurn(
                    text=f"I need a tool for this. Searching for: {need}.",
                    tool_calls=[
                        ToolCall(
                            id="call_find_tools",
                            name="find_tools",
                            arguments={"need": need},
                        )
                    ],
                )
            return ModelTurn(
                text=(
                    "The tool needed for this request is not available. "
                    "I cannot complete the action."
                )
            )

        if expedite:
            return ModelTurn(
                text=(
                    "I detected an exception on this purchase order (supplier cannot "
                    "ship on time). Proposed action: expedite_order. This action needs "
                    "human approval before I execute."
                ),
                tool_calls=[
                    ToolCall(
                        id="call_expedite",
                        name="expedite_order",
                        arguments={"order_id": order_id},
                    )
                ],
            )
        return ModelTurn(
            text=(
                "Looking up the purchase order status and details. "
                "This is a read-only action."
            ),
            tool_calls=[
                ToolCall(
                    id="call_lookup",
                    name="lookup_order",
                    arguments={"order_id": order_id},
                )
            ],
        )

    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        return self.complete_turn(messages, tools).text


def _extract_order_id(text: str) -> str:
    match = _ORDER_RE.search(text)
    return match.group(1).upper() if match else "PO-1234"


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


class OpenAIChatProvider:
    """Thin wrapper over the OpenAI-compatible chat completions API.

    Works with OpenAI, Azure OpenAI (base_url), DeepSeek, and Qwen/DashScope in most cases —
    they expose an OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url

    def complete_turn(
        self, messages: list[Message], tools: list[Tool]
    ) -> ModelTurn:
        try:
            import openai  # pyright: ignore[reportMissingImports]  # lazy optional dependency
        except ImportError as exc:
            raise ImportError(
                "OpenAIChatProvider requires the optional 'openai' package. "
                "Install it with: pip install 'sora-agent[openai]'"
            ) from exc

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in messages],
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.schema} for tool in tools
            ]
        resp = client.chat.completions.create(**payload)
        message = resp.choices[0].message

        text = message.content or ""
        if not isinstance(text, str):
            text = str(text)

        calls: list[ToolCall] = []
        for index, raw_call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(raw_call, "function", None)
            raw_arguments = getattr(function, "arguments", "") or "{}"
            arguments, error = _parse_arguments(raw_arguments)
            calls.append(
                ToolCall(
                    id=getattr(raw_call, "id", "") or f"call_{index}",
                    name=getattr(function, "name", "") or "",
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                    error=error,
                )
            )
        return ModelTurn(text=text, tool_calls=calls)

    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        return self.complete_turn(messages, tools).text


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.role == "assistant" and message.tool_calls:
        if not message.content:
            payload["content"] = None
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, default=str),
                },
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    return payload


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    if isinstance(raw, dict):
        return raw, ""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {}, f"invalid JSON arguments: {exc}"
    if not isinstance(value, dict):
        return {}, "tool arguments must be a JSON object"
    return value, ""


def default_provider(kind: str = "mock") -> ModelProvider:
    kind = kind.lower()
    if kind in {"mock", "demo"}:
        return MockProvider()
    if kind in {"openai", "azure"}:
        return OpenAIChatProvider(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if kind in {"deepseek", "ds"}:
        return OpenAIChatProvider(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    if kind in {"qwen", "dashscope"}:
        return OpenAIChatProvider(
            model=os.getenv("QWEN_MODEL", "qwen-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    raise ValueError(f"unknown provider kind: {kind}")
