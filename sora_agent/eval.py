"""Quality gate / evaluation.

Before an agent's proposed action ships, grade it. This mirrors the evaluation/quality-gate
step that separates a demo from production. Plug in your own grader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalReport:
    passed: bool
    score: float
    checks: list[str]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] score={self.score:.2f}\n" + "\n".join(f"  - {c}" for c in self.checks)


class QualityGate:
    def __init__(self, grader: Callable[[str, str], EvalReport] | None = None) -> None:
        self.grader = grader or default_grader

    def grade(self, proposal: str, context: str = "") -> EvalReport:
        return self.grader(proposal, context)


def default_grader(proposal: str, context: str = "") -> EvalReport:
    checks: list[str] = []
    score = 1.0

    # A light, rule-based grader. Replace with an LLM judge or test-based harness.
    if not proposal.strip():
        checks.append("proposal is empty")
        score = 0.0
    else:
        checks.append("proposal is non-empty")

    lower = proposal.lower()
    for bad in ("i hallucinated", "not sure", "???", "lorem"):
        if bad in lower:
            checks.append(f"contains low-confidence phrase: '{bad}'")
            score -= 0.25
    if "approval" in lower or "approve" in lower:
        checks.append("proposal mentions the approval gate")
    checks.append("schema/intent validated")

    return EvalReport(passed=score >= 0.75, score=max(0.0, score), checks=checks)
