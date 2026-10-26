"""A business state machine for purchase-order (PO) exception handling.

This is the concrete pattern enterprise copilots need: model the workflow as a state machine,
transition on business events, and only reach terminal states after human approval.

States: open -> deferred | expedited | cancelled | rerouted | exception
        any approved branch -> executed -> closed
        rejected -> back to open (and recorded for audit)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Set


class State(str, Enum):
    OPEN = "open"
    DEFERRED = "deferred"
    EXPEDITED = "expedited"
    CANCELLED = "cancelled"
    REROUTED = "rerouted"
    EXCEPTION = "exception"
    EXECUTED = "executed"
    CLOSED = "closed"


TRANSITIONS: dict[State, Set[State]] = {
    State.OPEN: {State.DEFERRED, State.EXPEDITED, State.CANCELLED, State.REROUTED, State.EXCEPTION},
    State.DEFERRED: {State.EXECUTED},
    State.EXPEDITED: {State.EXECUTED},
    State.CANCELLED: {State.EXECUTED},
    State.REROUTED: {State.EXECUTED},
    State.EXCEPTION: {State.EXECUTED},
    State.EXECUTED: {State.CLOSED},
    State.CLOSED: set(),
}


class InvalidTransition(Exception):
    pass


class POStateMachine:
    def __init__(self, po_id: str) -> None:
        self.po_id = po_id
        self.state = State.OPEN
        self.history: List[tuple[State, State, str]] = []

    def transition(self, target: State, reason: str = "") -> State:
        if target not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state} -> {target} is not allowed for {self.po_id}")
        src = self.state
        self.state = target
        self.history.append((src, target, reason))
        return self.state

    @property
    def is_terminal(self) -> bool:
        return not TRANSITIONS[self.state]

    def __repr__(self) -> str:
        return f"POStateMachine(po_id={self.po_id}, state={self.state.value})"
