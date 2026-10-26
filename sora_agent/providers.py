"""Model provider abstraction.

The agent loop never talks to a vendor directly. Swap the provider and everything else works.
Use `MockProvider` to run demos/CIs offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .tools import Tool


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


class ModelProvider(Protocol):
    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        """Return the assistant's text response."""
        ...


class MockProvider:
    """Returns canned text so the demo runs without an API key.

    It introspects the latest user message so it can look vaguely smart enough to demo
    approval + state transitions.
    """

    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        low = last_user.lower()
        if "expedite" in low or "stuck" in low or "can't ship" in low:
            return (
                "I detected an exception on this purchase order (supplier cannot ship on time). "
                "Proposed action: expedited.\n"
                "ACTION: expedite_order(order_id=\"PO-1234\")\n"
                "This action needs human approval before I execute."
            )
        if "approve" in low:
            return "Approved. Executing expedite via the order system now. Moving PO to closed."
        return ("Analyzing the request. I propose an action and will ask for approval before executing.\n"
                "ACTION: lookup_order(order_id=\"PO-1234\")")


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

    def complete(self, messages: list[Message], tools: list[Tool]) -> str:
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
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": tool.schema} for tool in tools]
        resp = client.chat.completions.create(**payload)
        return resp.choices[0].message.content or ""


def default_provider(kind: str = "mock") -> ModelProvider:
    kind = kind.lower()
    if kind in {"mock", "demo"}:
        return MockProvider()
    if kind in {"openai", "azure"}:
        return OpenAIChatProvider(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                  base_url=os.getenv("OPENAI_BASE_URL"))
    if kind in {"deepseek", "ds"}:
        return OpenAIChatProvider(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                                  api_key=os.getenv("DEEPSEEK_API_KEY"),
                                  base_url="https://api.deepseek.com")
    if kind in {"qwen", "dashscope"}:
        return OpenAIChatProvider(model=os.getenv("QWEN_MODEL", "qwen-plus"),
                                  api_key=os.getenv("DASHSCOPE_API_KEY"),
                                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    raise ValueError(f"unknown provider kind: {kind}")
