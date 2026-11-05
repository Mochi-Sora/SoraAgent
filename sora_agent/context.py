"""A small, deterministic context-window budget.

No tokenizer is bundled with the standard library, so this uses the common
``characters / 4`` estimate.  It is deliberately conservative: the goal is to
keep a conversation from growing without bound, not to be a billing meter.
"""

from __future__ import annotations

import json

from .providers import Message

DEFAULT_RESERVE_FRACTION = 0.2
MIN_RESERVE_TOKENS = 512


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _message_tokens(message: Message) -> int:
    tokens = 4 + estimate_tokens(message.content)
    if message.tool_calls:
        try:
            payload = json.dumps(
                [
                    {"name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ],
                default=str,
            )
        except (TypeError, ValueError):
            payload = ""
        tokens += estimate_tokens(payload)
    return tokens


def trim_messages(
    messages: list[Message],
    max_tokens: int,
    *,
    fixed_texts: tuple[str, ...] = (),
    reserve_tokens: int | None = None,
) -> list[Message]:
    """Return the newest messages that fit in ``max_tokens``.

    ``fixed_texts`` are strings that must also fit in the same budget, usually
    the system prompt and the incoming user message.  The newest message is
    always retained, truncated if it alone exceeds the remaining budget.
    """
    if max_tokens <= 0 or not messages:
        return list(messages)

    fixed = sum(estimate_tokens(text) for text in fixed_texts)
    if reserve_tokens is None:
        reserve_tokens = max(
            MIN_RESERVE_TOKENS,
            int(max_tokens * DEFAULT_RESERVE_FRACTION),
        )
    budget = max(256, max_tokens - fixed - reserve_tokens)

    kept: list[Message] = []
    used = 0
    for message in reversed(messages):
        cost = _message_tokens(message)
        if used + cost <= budget:
            kept.append(message)
            used += cost
            continue
        if kept:
            # Older messages do not fit; stop at the first contiguous recent
            # block so the conversation stays coherent.
            break
        # The newest message alone is too large: keep a truncated version.
        remaining_chars = max(64, (budget - 4) * 4)
        content = message.content
        if len(content) > remaining_chars:
            content = content[:remaining_chars] + "\n[...truncated to fit context]"
        kept.append(
            Message(
                role=message.role,
                content=content,
                tool_calls=list(message.tool_calls),
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
        )
        used = budget
        break

    kept.reverse()
    return kept
