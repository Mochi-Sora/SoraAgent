"""A Microsoft-ecosystem entrypoint: an Azure Function that wraps the agent harness.

This mirrors how you'd expose an agent behind Copilot Studio / Power Platform / Azure
Functions. The function takes a request, runs the agent loop (with its quality + approval
gates), and returns the result. In production the approval gate would notify a human
over Teams/Slack/email instead of reading stdin.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Make `python examples/copilot_function.py` work even without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sora_agent import Agent
from sora_agent.approval import ApprovalGate
from sora_agent.eval import QualityGate
from sora_agent.providers import default_provider
from sora_agent.state_machine import POStateMachine
from sora_agent.tools import registry

from _demo_memory import DemoMemory


def run_copilot_step(user_message: str, po_id: str = "PO-1234", auto_approve: bool = False) -> dict[str, Any]:
    """Entrypoint for an HTTP trigger / Copilot Studio action / Power Automate step."""

    provider = default_provider(os.getenv("AGENT_PROVIDER", "mock"))
    approval = ApprovalGate()
    if auto_approve:
        approval.request = lambda action, detail: approval.approve_for(action, detail)  # type: ignore[method-assign]

    agent = Agent(
        provider=provider,
        registry=registry,
        approval=approval,
        quality=QualityGate(),
        memory=DemoMemory(),
    )
    sm = POStateMachine(po_id=po_id)

    task = f"{user_message}\nPO id: {po_id}"
    result = agent.run(task, state=sm)

    return {
        "proposal": result.proposal,
        "final_answer": result.final_answer,
        "stop_reason": result.stop_reason,
        "quality": str(result.report),
        "action": result.action,
        "tool_result": result.tool_result,
        "state": str(sm),
        "approved": result.approved,
        "audit": [str(a) for a in approval.audit],
        "steps": [
            {
                "visible": step.tools,
                "calls": [item.call.name for item in step.results],
            }
            for step in result.steps
        ],
    }


if __name__ == "__main__":
    # Simulate the HTTP trigger locally.
    import json

    out = run_copilot_step(
        "Purchase order PO-1234 is stuck; the supplier can't ship on time. Expedite it.",
        auto_approve=True,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
