"""A minimal chat handler: keep per-session history, run the model/tool loop."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from ..agent import MEMORY_POLICY, TOOLS_POLICY
from ..eval import QualityGate
from ..providers import Message
from ..runner import RunResult
from ..runtime import AgentRuntime, RunRequest, wrap_identity
from ..tool_search import FIND_TOOLS_NAME
from ..tools import ToolRegistry
from .approval import ApprovalBroker
from .types import InboundMessage

log = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = (
    "You are a careful assistant talking with a user over a chat channel. "
    "Answer directly and concisely. Use the tools available to you when they "
    "help, and never claim an action succeeded unless a tool confirmed it."
)


@dataclass
class ChatState:
    """The little bit of run state the memory hooks read."""

    session_id: str
    state: str = ""


class ChatHandler:
    """Turn inbound chat messages into replies.

    The gateway runs this from a worker thread, but calls are serialized by a
    lock so one session's history cannot interleave with another's.
    """

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry | None = None,
        *,
        memory: Any = None,
        identity: Any = None,
        approvals: ApprovalBroker | None = None,
        quality: QualityGate | None = None,
        max_steps: int = 8,
        max_turns: int = 40,
        max_tool_result_chars: int = 4000,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()
        if self.registry.has(FIND_TOOLS_NAME):
            raise ValueError(
                f"the tool name {FIND_TOOLS_NAME!r} is reserved for lazy "
                "tool discovery"
            )
        self.memory = memory
        self.identity = None if identity is False else identity
        self.approvals = approvals
        self.quality = quality or QualityGate()
        self.max_steps = max(1, int(max_steps))
        self.max_turns = max(0, int(max_turns))
        self.max_tool_result_chars = max(0, int(max_tool_result_chars))
        self.system_prompt = system_prompt
        if self.identity is not None:
            self._attach_identity(self.identity, memory)
            self.provider = wrap_identity(self.identity, provider)
        self.memory_tool: Any = None
        if memory is not None:
            try:
                self.memory_tool = memory.tool()
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory tool registration failed: %s", exc)
        self._history: dict[str, list[Message]] = {}
        self._lock = threading.Lock()

    def handle(self, inbound: InboundMessage) -> str:
        """Run one turn and return the text to send back."""
        key = inbound.session_key
        with self._lock:
            history = list(self._history.get(key, ()))

        approval = (
            self.approvals.gate(inbound) if self.approvals is not None else None
        )
        messages = [
            Message("system", self._prompt()),
            *history,
            Message("user", inbound.content),
        ]
        state = ChatState(session_id=key)
        self._notify("before_run", inbound.content, state)
        run = AgentRuntime.run(
            RunRequest(
                provider=self.provider,
                registry=self.registry,
                messages=messages,
                memory_tool=self.memory_tool,
                approval=approval,
                quality=self.quality,
                max_steps=self.max_steps,
                max_result_chars=self.max_tool_result_chars,
                context=f"channel={inbound.channel}",
            )
        )
        self._notify("after_run", run, state)

        reply = _reply_text(run)
        with self._lock:
            stored = self._history.setdefault(key, [])
            stored.append(Message("user", inbound.content))
            stored.append(Message("assistant", reply))
            limit = self.max_turns * 2
            if limit and len(stored) > limit:
                del stored[: len(stored) - limit]
        return reply

    def reset(self, session_key: str) -> None:
        """Forget one conversation."""
        with self._lock:
            self._history.pop(session_key, None)

    def _prompt(self) -> str:
        prompt = self.system_prompt + TOOLS_POLICY
        if self.memory_tool is not None:
            prompt += MEMORY_POLICY
        return prompt

    @staticmethod
    def _attach_identity(identity: Any, memory: Any) -> None:
        setter = getattr(memory, "set_identity", None)
        if not callable(setter):
            return
        try:
            setter(identity)
        except Exception as exc:  # noqa: BLE001 - identity is optional
            log.warning("memory identity attachment failed: %s", exc)

    def _notify(self, name: str, *args: Any) -> None:
        if self.memory is None:
            return
        callback = getattr(self.memory, name, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:  # noqa: BLE001 - memory must fail open
            log.warning("memory %s failed: %s", name, exc)


def _reply_text(run: RunResult) -> str:
    text = run.final_answer.strip()
    if text:
        return text
    if run.stop_reason == "approval_rejected":
        return (
            "I didn't run that: it needs human approval, and none was given."
        )
    for step in reversed(run.steps):
        candidate = step.turn.text.strip()
        if candidate:
            return candidate
    if run.error:
        return f"I couldn't finish that: {run.error}"
    return "I couldn't produce a reply for that message."
