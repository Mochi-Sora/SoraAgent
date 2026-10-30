"""End-to-end demo: PO exception handled by an agent with a human-approval gate.

Run interactively:
    python examples/po_workflow.py

Run without prompting (auto-approve):
    python examples/po_workflow.py --yes

Real model:
    export OPENAI_API_KEY=... && python examples/po_workflow.py --provider openai
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `python examples/po_workflow.py` work even without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _demo_memory import DemoMemory

from sora_agent import Agent, MockProvider, POStateMachine
from sora_agent.approval import ApprovalGate
from sora_agent.providers import default_provider
from sora_agent.tools import registry


class AutoApproveGate(ApprovalGate):
    def request(self, action: str, detail: str) -> bool:
        return self.approve_for(action, detail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--yes", action="store_true", help="auto-approve (non-interactive)")
    args = ap.parse_args()

    provider = MockProvider() if args.provider == "mock" else default_provider(args.provider)
    gate = AutoApproveGate() if args.yes else ApprovalGate()

    agent = Agent(
        provider=provider,
        registry=registry,
        approval=gate,
        memory=DemoMemory(),
    )
    sm = POStateMachine(po_id="PO-1234")

    task = (
        f"Purchase order {sm.po_id} is stuck. The supplier can't ship on time. "
        "Expedite it so the customer isn't delayed."
    )

    print(f"\nSTATE BEFORE: {sm}")
    result = agent.run(task, state=sm)

    print("\n--- ANSWER -------------------------------------------")
    print(result.final_answer or result.proposal)
    print(f"\nstop_reason={result.stop_reason}")
    print("\n--- STEPS --------------------------------------------")
    for step in result.steps:
        calls = ", ".join(item.call.name for item in step.results)
        print(
            f"  {step.index}: visible={', '.join(step.tools)}"
            + (f"; calls={calls}" if calls else "")
        )
    print("\n--- QUALITY GATE -------------------------------------")
    print(result.report)
    print("\n--- TOOL RESULT ---------------------------------------")
    print(result.tool_result)
    print("\n--- STATE AFTER ---------------------------------------")
    print(f"{sm}  (terminal={sm.is_terminal})")
    print("\n--- AUDIT TRAIL ---------------------------------------")
    for entry in gate.audit:
        print("  " + str(entry))


if __name__ == "__main__":
    main()
