# ``tools`` imports ``ToolCall`` from this module, so the tool import below is
# type-checking-only.  The static cycle is real but cannot occur at runtime.
# pyright: reportImportCycles=false
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Union
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .tools import Tool


_ACTION_RE = re.compile(
    r"ACTION:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", re.DOTALL
)
_ORDER_RE = re.compile(r"\b(PO-\d+)\b", re.IGNORECASE)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ProviderConfigError(RuntimeError):
    """A provider is missing configuration, usually its API key."""


def _is_loopback(host: str | None) -> bool:
    value = (host or "").strip("[]").lower()
    return value in _LOOPBACK_HOSTS or value.startswith("127.")


def _check_base_url(url: str | None) -> None:
    """Reject unusable or plaintext remote endpoints."""
    if not url:
        return
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ProviderConfigError(
            "provider base_url must start with http:// or https://"
        )
    if parsed.scheme != "https" and not _is_loopback(parsed.hostname):
        raise ProviderConfigError(
            "refusing to send credentials or chat content over plain HTTP to "
            f"{parsed.hostname or url!r}; use https:// or a loopback address"
        )


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
class TokenUsage:
    """Provider-reported token counts for one turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelTurn:
    """One assistant turn: visible text plus any structured tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None


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
    *,
    on_delta: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ModelTurn:
    """Ask any provider for a structured turn.

    Providers with ``stream_turn`` stream chunks through ``on_delta`` when a
    consumer asks for them; everyone else is called once and their whole text is
    emitted as a single chunk.  Providers with ``complete_turn`` return
    structured tool calls.  Providers with only ``complete`` get their text
    parsed for a single legacy ``ACTION:`` line.  This is the only place where
    the text protocol is understood.
    """
    if on_delta is not None:
        stream = getattr(provider, "stream_turn", None)
        if callable(stream):
            return _coerce_turn(
                stream(
                    list(messages),
                    list(tools),
                    on_delta,
                    should_stop=should_stop,
                )
            )

    method = getattr(provider, "complete_turn", None)
    if callable(method):
        turn = _coerce_turn(method(list(messages), list(tools)))
        if (
            on_delta is not None
            and turn.text
            and (should_stop is None or not should_stop())
        ):
            on_delta(turn.text)
        return turn

    complete = getattr(provider, "complete", None)
    if not callable(complete):
        raise TypeError(
            f"provider {type(provider).__name__} has neither complete_turn() "
            "nor complete()"
        )
    turn = turn_from_text(complete(list(messages), list(tools)))
    if (
        on_delta is not None
        and turn.text
        and (should_stop is None or not should_stop())
    ):
        on_delta(turn.text)
    return turn


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

    def stream_turn(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_delta: Callable[[str], None],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ModelTurn:
        """Offline stand-in for token streaming: emit word-sized chunks."""
        turn = self.complete_turn(messages, tools)
        emit_words(turn.text, on_delta, should_stop)
        return turn


def emit_words(
    text: str,
    on_delta: Callable[[str], None],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Emit text in word-sized chunks for providers that cannot stream."""
    if not text:
        return
    for match in re.finditer(r"\S+\s*", text):
        if should_stop is not None and should_stop():
            return
        on_delta(match.group(0))


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
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        stream_usage: bool = True,
        temperature: float | None = None,
        require_api_key: bool = True,
    ) -> None:
        self.model = model
        self.api_key = (api_key or "").strip()
        if require_api_key and not self.api_key:
            raise ProviderConfigError(
                "an API key is required for OpenAIChatProvider"
            )
        _check_base_url(base_url)
        self.base_url = base_url
        self.timeout = timeout
        self.stream_usage = stream_usage
        if temperature is not None and not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        self.temperature = (
            None if temperature is None else float(temperature)
        )

    def complete_turn(
        self, messages: list[Message], tools: list[Tool]
    ) -> ModelTurn:
        client = self._client()
        response = client.chat.completions.create(**self._payload(messages, tools))
        message = response.choices[0].message

        text = message.content or ""
        if not isinstance(text, str):
            text = str(text)

        return ModelTurn(
            text=text,
            tool_calls=_tool_calls(getattr(message, "tool_calls", None) or []),
            usage=_usage_from(getattr(response, "usage", None)),
        )

    def stream_turn(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_delta: Callable[[str], None],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ModelTurn:
        """Stream tokens through ``on_delta`` and reassemble tool calls."""
        client = self._client()
        payload = self._payload(messages, tools)
        payload["stream"] = True
        if self.stream_usage:
            payload["stream_options"] = {"include_usage": True}
        stream = client.chat.completions.create(**payload)
        parts: list[str] = []
        fragments: dict[int, dict[str, str]] = {}
        usage: TokenUsage | None = None
        try:
            for chunk in stream:
                if should_stop is not None and should_stop():
                    break
                chunk_usage = _usage_from(getattr(chunk, "usage", None))
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    piece = str(content)
                    parts.append(piece)
                    on_delta(piece)
                for raw_call in getattr(delta, "tool_calls", None) or []:
                    _merge_call(fragments, raw_call)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        return ModelTurn(
            text="".join(parts),
            tool_calls=_fragment_calls(fragments),
            usage=usage,
        )

    def _client(self) -> Any:
        try:
            import openai  # pyright: ignore[reportMissingImports]  # lazy optional dependency
        except ImportError as exc:
            raise ImportError(
                "OpenAIChatProvider requires the optional 'openai' package. "
                "Install it with: pip install 'sora-agent[openai]'"
            ) from exc
        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _payload(
        self, messages: list[Message], tools: list[Tool]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in messages],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.schema} for tool in tools
            ]
        return payload

    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        return self.complete_turn(messages, tools).text


def _usage_from(raw: Any) -> TokenUsage | None:
    if raw is None:
        return None
    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion = int(getattr(raw, "completion_tokens", 0) or 0)
    total = int(getattr(raw, "total_tokens", 0) or 0)
    if not (prompt or completion or total):
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total or prompt + completion,
    )


def _tool_calls(raw_calls: Sequence[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
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
    return calls


def _merge_call(fragments: dict[int, dict[str, str]], raw: Any) -> None:
    index = getattr(raw, "index", 0)
    key = index if isinstance(index, int) else 0
    entry = fragments.setdefault(key, {"id": "", "name": "", "arguments": ""})
    call_id = getattr(raw, "id", None)
    if call_id:
        entry["id"] = str(call_id)
    function = getattr(raw, "function", None)
    if function is None:
        return
    name = getattr(function, "name", None)
    if name:
        entry["name"] = str(name)
    arguments = getattr(function, "arguments", None)
    if arguments:
        entry["arguments"] += str(arguments)


def _fragment_calls(fragments: dict[int, dict[str, str]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(fragments):
        entry = fragments[index]
        raw_arguments = entry["arguments"] or "{}"
        arguments, error = _parse_arguments(raw_arguments)
        calls.append(
            ToolCall(
                id=entry["id"] or f"call_{index}",
                name=entry["name"],
                arguments=arguments,
                raw_arguments=raw_arguments,
                error=error,
            )
        )
    return calls


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


def default_provider(
    kind: str = "mock",
    timeout: float = 30.0,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> ModelProvider:
    """Build a provider from explicit values and environment defaults.

    Explicit arguments win, then the environment, then the built-in defaults
    for that provider kind.  A real provider without a key raises
    :class:`ProviderConfigError`; keys are never borrowed from a different
    provider's environment variable.
    """
    kind = kind.lower()
    if kind in {"mock", "demo"}:
        return MockProvider()
    if kind in {"openai", "azure"}:
        return OpenAIChatProvider(
            model=model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            api_key=_required_key("OpenAI", api_key, "OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
            temperature=temperature,
        )
    if kind in {"deepseek", "ds"}:
        return OpenAIChatProvider(
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            api_key=_required_key(
                "DeepSeek", api_key, "DEEPSEEK_API_KEY"
            ),
            base_url=(
                base_url
                or os.getenv("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com"
            ),
            timeout=timeout,
            temperature=temperature,
        )
    if kind in {"qwen", "dashscope"}:
        return OpenAIChatProvider(
            model=model or os.getenv("QWEN_MODEL", "qwen3.7-plus"),
            api_key=_required_key(
                "Qwen (DashScope)", api_key, "DASHSCOPE_API_KEY"
            ),
            base_url=(
                base_url
                or os.getenv("QWEN_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            timeout=timeout,
            temperature=temperature,
        )
    if kind in {"custom", "openai-compatible"}:
        if not base_url:
            raise ProviderConfigError(
                "custom provider requires a base_url (an OpenAI-compatible "
                "chat completions endpoint)"
            )
        return OpenAIChatProvider(
            model=model or os.getenv("SORA_CUSTOM_MODEL", "gpt-5.4-mini"),
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            require_api_key=False,
        )
    raise ValueError(f"unknown provider kind: {kind}")


def _required_key(
    label: str, explicit: str | None, env_name: str
) -> str:
    value = (explicit or os.getenv(env_name, "")).strip()
    if not value:
        raise ProviderConfigError(
            f"{label} requires an API key: set {env_name} or pass api_key"
        )
    return value


_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})
_RETRYABLE_NAME_HINTS = (
    "timeout",
    "connection",
    "ratelimit",
    "server",
    "unavailable",
    "overloaded",
    "internalserver",
)
_PERMANENT_NAME_HINTS = (
    "authentication",
    "permission",
    "invalidrequest",
    "badrequest",
    "notfound",
    "unprocessable",
    "contentpolicy",
)


def is_retryable_error(exc: BaseException) -> bool:
    """Classify a provider failure as transient or permanent.

    Authentication, permission, validation, and configuration failures will
    not improve by retrying.  Timeouts, connection failures, rate limits, and
    5xx responses will.  Unknown errors keep the historical retry behaviour.
    """
    if isinstance(
        exc,
        (ProviderConfigError, ValueError, TypeError, KeyError, ImportError),
    ):
        return False
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        if status in _RETRYABLE_STATUS or 500 <= status <= 599:
            return True
        if 400 <= status <= 499:
            return False
    name = type(exc).__name__.casefold()
    if any(hint in name for hint in _PERMANENT_NAME_HINTS):
        return False
    if any(hint in name for hint in _RETRYABLE_NAME_HINTS):
        return True
    return True
