"""Optional adapter that makes any ModelProvider state-aware.

This is the only file that connects the self-contained state subsystem to the
rest of SoraAgent. It wraps a provider, appends the session state to the *end*
of the system prompt (preserving the static prefix for prompt caching), extracts
state edits from responses, and runs the bounded deterministic repair loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .providers import (
    Message,
    ModelProvider,
    ModelTurn,
    provider_text,
    provider_turn,
)
from .state import (
    StateIssue,
    StateManager,
    extract_state_edit,
)


class StatefulProvider:
    """Wrap a provider with persistent session state and validation."""

    def __init__(self, inner: ModelProvider, state: StateManager) -> None:
        self.inner = inner
        self.state = state

    def complete_turn(
        self, messages: list[Message], tools: list[Any]
    ) -> ModelTurn:
        augmented = self._augment(messages)
        turn = provider_turn(self.inner, augmented, tools)
        text = self._settle(turn.text, augmented, tools)
        return ModelTurn(
            text=text,
            tool_calls=list(turn.tool_calls),
            usage=turn.usage,
        )

    def complete(self, messages: list[Message], tools: list[Any]) -> str:
        augmented = self._augment(messages)
        raw = provider_text(self.inner, augmented, tools)
        return self._settle(raw, augmented, tools)

    def stream_turn(
        self,
        messages: list[Message],
        tools: list[Any],
        on_delta: Callable[[str], None],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ModelTurn:
        augmented = self._augment(messages)
        stream = getattr(self.inner, "stream_turn", None)
        if callable(stream):
            turn = cast(
                ModelTurn,
                stream(augmented, tools, on_delta, should_stop=should_stop),
            )
        else:
            turn = provider_turn(self.inner, augmented, tools)
            if turn.text:
                on_delta(turn.text)
        text = self._settle(turn.text, augmented, tools)
        return ModelTurn(
            text=text,
            tool_calls=list(turn.tool_calls),
            usage=turn.usage,
        )

    def _augment(self, messages: list[Message]) -> list[Message]:
        goal = next(
            (
                message.content
                for message in reversed(messages)
                if getattr(message, "role", "") == "user"
            ),
            "",
        )
        current = self.state.ensure(goal)
        return self._with_state(
            messages, self.state.prompt_section(current)
        )

    def _settle(
        self,
        raw: str,
        base_messages: list[Message],
        tools: list[Any],
    ) -> str:
        parsed = extract_state_edit(raw)
        if not parsed.issues and parsed.edit is None:
            return raw

        # The state subsystem must never rewrite the agent's visible answer.
        # Keep the original text (with the state block removed) as the only
        # value returned to the agent loop.
        visible = parsed.text
        issues: list[StateIssue] = list(parsed.issues)

        if parsed.edit is not None and not parsed.issues:
            applied = self.state.apply(parsed.edit)
            if applied.status in {"accepted", "unchanged"}:
                return visible
            issues = list(applied.validation.issues)

        for attempt in range(1, self.state.config.max_repairs + 1):
            repair = self.state.build_repair_prompt(
                issues, self.state.ensure(), attempt
            )
            repair_messages = list(base_messages) + [
                Message("assistant", raw),
                Message("user", repair),
            ]
            raw = provider_text(self.inner, repair_messages, tools)
            parsed = extract_state_edit(raw)

            if parsed.issues:
                issues = list(parsed.issues)
                continue
            if parsed.edit is None:
                issues = [
                    StateIssue(
                        "E_PROTOCOL",
                        "No state_edit block was returned.",
                    )
                ]
                continue

            applied = self.state.apply(parsed.edit)
            if applied.status in {"accepted", "unchanged"}:
                return visible
            issues = list(applied.validation.issues)

        return visible

    @staticmethod
    def _with_state(
        messages: list[Message], section: str
    ) -> list[Message]:
        """Insert the state as the final system message.

        The original system messages are left byte-for-byte untouched so their
        cached prefix remains valid. Later static system content can be added
        before this trailing state message.
        """
        augmented = list(messages)
        insert_at = 0
        for index, message in enumerate(augmented):
            if getattr(message, "role", "") == "system":
                insert_at = index + 1
            else:
                break
        augmented.insert(insert_at, Message("system", section))
        return augmented
