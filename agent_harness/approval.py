"""Human-approval gate + audit trail.

Some actions (expedite / cancel / move money) should never execute without a human saying yes.
This gate blocks, records who approved, and emits an audit entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AuditEntry:
    ts: float
    action: str
    actor: str
    decision: str
    detail: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.ts:.0f}] {self.action} actor={self.actor} "
            f"decision={self.decision} {self.detail}"
        )


class ApprovalGate:
    def __init__(self) -> None:
        self.audit: List[AuditEntry] = []

    def request(self, action: str, detail: str) -> bool:
        """In a real system this would push a notification to a human / Slack / Teams.

        For the harness we ask via stdin so demos are interactive. Return True to approve.
        """
        prompt = (
            f"\n=== HUMAN APPROVAL REQUIRED ===\n"
            f"Action: {action}\nDetail: {detail}\nApprove? [y/N] "
        )
        ans = input(prompt).strip().lower()
        decision = "approved" if ans.startswith("y") else "rejected"
        self.audit.append(
            AuditEntry(time.time(), action, "human-reviewer", decision, detail)
        )
        return decision == "approved"

    def approve_for(self, action: str, detail: str, actor: str = "human-reviewer") -> bool:
        """Programmatic approval (used by tests / non-interactive runs)."""
        self.audit.append(AuditEntry(time.time(), action, actor, "approved", detail))
        return True

    def log(self, action: str, decision: str, detail: str, actor: str = "agent") -> None:
        self.audit.append(AuditEntry(time.time(), action, actor, decision, detail))
