# ai-agent-scaffold

A minimal, model-agnostic **agent harness** you can start from. It's the scaffolding I wish I
had when building production agents — and it doubles as the open-source lead magnet on my
portfolio site.

> **What it teaches by example:** a clean agent loop, pluggable model providers, tool calling,
> a **human-approval gate** before consequential actions, a **quality gate / evaluation**
> before anything ships, a simple RAG helper, and a **state machine** for a real business
> workflow (order / PO exception handling).

---

## Why not a heavy framework?

Most agent frameworks lock you into one model vendor and hide the parts that matter in
production: **when to stop and ask a human**, **how to grade output quality**, and **how to
make the model swappable**. This repo keeps those three front and center and stays small enough
to read in an hour.

---

## Features

- **Pluggable providers** — `OpenAIProvider`, `AzureOpenAIProvider`, `DeepSeekProvider`,
  `QwenProvider`, and a `MockProvider` so you can run the demo with no API key.
- **Agent loop** — planner → tool call → **approval gate** → execute → observe → (rerun).
- **Human-approval gate** — any tool/action can be marked `needs_approval`; the loop blocks
  until a human confirms or rejects, and records the decision + actor (audit).
- **Quality gate / evaluation** — a pluggable grader that blocks a result that fails your bar
  (e.g. hallucination / missing-source / schema checks), with a pass/fail report.
- **RAG helper** — a `retrieve()` interface with an optional in-memory stub and clear hooks for
  a real vector DB.
- **State machine** — `POStateMachine` models `open → deferred / expedited / cancelled /
  rerouted / exception → approve → executed/closed`.
- **Tools** — a decorator-based registry; add a function and it becomes callable by the agent.

---

## Quickstart (no API key needed)

```bash
python -m pip install -e .
python examples/po_workflow.py            # uses MockProvider; prints the audit trail
```

With a real model:

```bash
export OPENAI_API_KEY=...                 # or DEEPSEEK_API_KEY / DASHSCOPE_API_KEY
python examples/po_workflow.py --provider openai
```

---

## Layout

```
ai-agent-scaffold/
  agent_harness/
    providers.py       # model provider abstraction + MockProvider
    agent.py           # the agent loop (plan → tool → gate → execute → observe)
    tools.py           # @tool registry + example tools
    approval.py        # human-approval gate + audit record
    eval.py            # quality gate / evaluation
    rag.py             # RAG retriever interface + in-memory stub
    state_machine.py   # PO / order exception state machine
  examples/
    po_workflow.py     # end-to-end demo with approval + audit
    copilot_function.py# Azure Function-style entrypoint (Microsoft ecosystem)
  pyproject.toml
```

---

## Example: state machine + approval

```python
from agent_harness.agent import Agent
from agent_harness.providers import MockProvider
from agent_harness.state_machine import POStateMachine

agent = Agent(provider=MockProvider())
sm = POStateMachine(po_id="PO-1234")

result = agent.run("Purchase order 1234 is stuck; the vendor can't ship on time. Expedite it.",
                   state=sm)

print(result.audit)   # every step + who approved
```

Run it and you'll see the agent detect the exception, propose `expedited`, **stop for human
approval**, then execute and only move to `closed` after approval — exactly the pattern
enterprise copilots need.

---

## Notes for the repo owner

- Add your real credentials via env vars / a `.env` kept out of git.
- Swap `MemoryRetriever` for a real vector store to turn `rag.py` into production RAG.
- The `MockProvider` returns canned text so CI / demos run offline; replace in prod.

### License
MIT. Fork it, build on it, and say hi — this is also how I like to work with clients.
