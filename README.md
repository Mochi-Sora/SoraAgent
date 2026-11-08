# SoraAgent

**A model-agnostic agent harness with human-approval and quality gates.**

SoraAgent is a small, standard-library-only Python framework for building
agentic applications on top of any chat model. The model/tool loop is
deliberately tiny; the production concerns live in explicit gates around it:

- **Approval gate** — tools marked `needs_approval` never execute without a
  human saying yes (stdin for CLI/library use, chat replies or buttons for
  channels).
- **Quality gate** — the finished trajectory of every `Agent` or chat run is
  graded by a deterministic structural evaluator that *cannot* affect execution.
- **Lazy tool discovery** — only `find_tools` (plus `memory_recall`) is visible
  at the start of a run; other tools stay hidden until the model previews and
  loads what it needs, keeping the prompt small and the tool list bounded.
- **Predictable failure modes** — safety paths (approval, tool validation) fail
  closed; optional conveniences (memory, identity, clock, calendar, observers)
  fail open and never break a run.
- **Model-facing subsystems** — a per-turn clock, a native calendar, a
  persistent file-backed identity (`USER.md` / `SOUL.md`), a validated session
  state document, and an optional CortexDB-backed long-term memory with daily
  verbatim recall and nightly consolidation ("dreams").
- **Chat channels** — ten in-box transports (Telegram, Discord, Slack,
  WhatsApp Cloud API, Microsoft Teams, Signal, Google Chat, Matrix,
  Feishu/Lark, and iMessage via BlueBubbles) with per-session history,
  human approvals rendered as chat prompts, and a "preparing a response"
  indicator on every platform that exposes one.

The core imports only the Python standard library; optional integrations
(`openai`, `zstandard`) are extras. Requires Python **3.9+**. Current version:
**0.1.0**.

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Architecture](#architecture)
5. [The model/tool loop (Runner)](#the-modeltool-loop-runner)
6. [Providers](#providers)
7. [Tools](#tools)
8. [Lazy tool discovery](#lazy-tool-discovery)
9. [Human approval](#human-approval)
10. [Quality gate and evaluation](#quality-gate-and-evaluation)
11. [Optional trajectory judge](#optional-trajectory-judge)
12. [Clock](#clock)
13. [Calendar](#calendar)
14. [Session state](#session-state)
15. [Identity (USER.md / SOUL.md)](#identity-usermd--soulmd)
16. [Memory (CortexDB)](#memory-cortexdb)
17. [Chat channels and the gateway](#chat-channels-and-the-gateway)
18. [Command-line interfaces](#command-line-interfaces)
19. [Environment variables](#environment-variables)
20. [Project layout](#project-layout)
21. [Development](#development)
22. [License](#license)

---

## Requirements

- Python **3.9+** (declared in `pyproject.toml`; static checks target 3.9).
- No mandatory third-party dependencies.
- Optional:
  - `openai>=1.0` for the OpenAI-compatible provider (`pip install 'sora-agent[openai]'`).
  - `zstandard>=0.22` for compressed memory backups on Python < 3.14
    (`pip install 'sora-agent[memory]'`). On Python 3.14+ the standard library
    `compression.zstd` module is used instead.
  - `typer`, `rich`, and `questionary` for the guided `sora` command
    (`pip install 'sora-agent[cli]'`). The core library never imports them.
  - `pypdf>=4` for PDF attachment text extraction
    (`pip install 'sora-agent[files]'`). Other attachment formats use the
    standard library; image description can use an OpenAI-compatible vision
    model or optional OCR (`pytesseract` + Pillow).
  - Telegram works over `urllib`; no `python-telegram-bot` dependency is needed.
    The other in-box channels use the same standard-library approach (Discord
    and Slack WebSockets, signed HTTP webhooks, or platform REST APIs). Signal
    requires a `signal-cli` daemon and iMessage requires a BlueBubbles server;
    both are external services, not Python dependencies.

## Installation

### One-line installer (recommended)

**macOS / Linux**

```bash
curl -fsSL https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.sh | bash
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.ps1 | iex
```

The installer needs Python **3.9+** already available and downloads the latest
release (falling back to `main` when no release exists). It installs SoraAgent
into an isolated virtual environment, adds the command directory to your PATH,
and verifies all console scripts. On Debian/Ubuntu systems without
`python3-venv`, it creates the environment without pip and bootstraps pip
inside it, so no `sudo` is required. After installation, run `sora update` to
upgrade in place; user data under `~/.sora_agent` is never touched. Re-running
the installer also works and is still required if the CLI extra was not
installed.

Uninstall removes only files the installer created: command wrappers must carry
its marker, and the install root must contain `venv/pyvenv.cfg`. Drive roots,
`$HOME`, ancestors of `$HOME`, and unrelated non-empty directories are refused.
Use `--purge` to remove `~/.sora_agent` data as well.

Common options:

```bash
# Core package only (no optional extras; no `sora` wizard)
curl -fsSL .../install.sh | SORA_EXTRAS=none bash

# Select a branch or release tag (the v prefix is optional)
curl -fsSL .../install.sh | bash -s -- --version main

# Install from a local checkout
./install.sh --source .

# Uninstall (add --purge to delete ~/.sora_agent data as well)
curl -fsSL .../install.sh | bash -s -- --uninstall
```

On Windows, set options through environment variables before `irm ... | iex`
(`$env:SORA_EXTRAS = "none"`), or invoke the downloaded script block with
parameters:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.ps1))) -Version main
```

| Option | Environment variable | Default |
| --- | --- | --- |
| release tag or `main` | `SORA_VERSION` | latest release, else `main` |
| extras (`openai,memory,cli,files`, ...) or `none` | `SORA_EXTRAS` | `openai,memory,cli,files` |
| install root | `SORA_INSTALL_DIR` | `~/.local/share/sora-agent` (`%LOCALAPPDATA%\SoraAgent` on Windows) |
| command directory | `SORA_BIN_DIR` | `~/.local/bin` (`<install root>\bin` on Windows) |
| Python executable | `SORA_PYTHON` | first Python 3.9+ found |
| local source checkout | `SORA_SOURCE` | — |
| fork/repository | `SORA_REPO` | `Mochi-Sora/SoraAgent` |

### From a checkout (contributors)

```bash
git clone https://github.com/Mochi-Sora/SoraAgent.git
cd SoraAgent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Core harness (no dependencies)
pip install -e .

# Optional extras
pip install -e ".[openai,memory,cli,files,dev]"
```

Both install routes create three console scripts:

| Command | Module | Purpose |
| --- | --- | --- |
| `sora` | `sora_agent.cli:main` | Guided setup, terminal chat, one-shot runs, gateway, config, doctor |
| `sora-gateway` | `sora_agent.channels.cli:main` | Run the agent behind a chat channel |
| `sora-memory` | `sora_agent.memory.cli:main` | Operate the memory subsystem |

All three also run as modules: `python -m sora_agent.cli`,
`python -m sora_agent.channels`, and `python -m sora_agent.memory`.

### Updating an existing install

```bash
sora update              # latest release (fails closed if the release lookup fails)
sora update --check      # only report whether an update is available
sora update --version main
sora update --extras none
sora update --extras openai,memory,cli,files   # add PDF support to an old install
```

`sora update` detects how SoraAgent was installed:

- **Installer-managed environment:** downloads the selected GitHub archive and
  reinstalls it into the current virtual environment. `cli` and `files` are
  always kept; `openai` and `memory` are detected from the installed packages.
- **Editable or source checkout:** pulls the current branch with `--ff-only`,
  then reinstalls in place. A checkout with uncommitted changes is never
  modified; commit or stash first. `--version TAG` checks out an explicit tag
  and leaves the checkout in detached-HEAD state.

Installs created before the `files` extra existed (`pypdf`) gain PDF support on
`update`. Because the update runs the updater that is already installed, such an
install may need `sora update` twice: the first run installs the new code, the
second applies the current defaults. Passing
`--extras openai,memory,cli,files` does it in one go.

`--check` exits with `0` when the install is current, `1` when an update is
available, and `2` when the check cannot run. User data and configuration under
`~/.sora_agent` are not touched.

## Quick start

### Set up Sora with the wizard

The `sora` command configures the gateway, model provider, memory, identity,
tools, approvals, and quality evaluation in one place:

```bash
sora setup
```

The first screen offers exactly two paths:

| Mode | What it does |
| --- | --- |
| **Recommended Setup** | Everything on and tuned for daily use. Asks only for the channel, main model, agent name/icon, temperature, and personality, then shows a review you can edit. |
| **Advanced Setup (tinker with caution)** | Every optional subsystem starts **off**, including memory, file understanding, identity, clock, calendar, sessions, tools, approval prompts, and quality evaluation. Each section can be edited individually and ends with an explicit save confirmation. |

Recommended Setup writes full long-term memory (including the CortexDB
sidecar), file understanding, identity, clock, calendar, sessions, the demo
tool pack, approval prompts, quality evaluation, and limits of 12
steps / 80 turns / 60 s timeouts / 3 retries.

Advanced Setup starts with temperature `0.1` and safety features disabled. It
shows warning panels and requires typing `I understand` before continuing and
`save` before writing the file. With approval off, consequential tools execute
without asking (an audit entry is still recorded). With quality off, `/eval`
reports that evaluation is disabled. Low-level validation, timeouts, process
isolation, and audit logging always remain active.

After setup:

```bash
sora chat                              # local terminal chat
sora run "expedite PO-1234"            # one task, then exit
sora gateway start                     # chat gateway in the background
sora doctor                            # configuration and safety check
```

`SORA_CONFIG` and `SORA_SECRETS` can point at different files; the defaults are
`~/.sora_agent/config.json` and `~/.sora_agent/secrets.env`. Both are written
atomically with mode `0600`, and the directory is mode `0700`. The config file
never contains API keys or bot tokens. Precedence is always:

```
command-line flag  >  environment variable  >  config file  >  built-in default
```

### An offline agent with approval and a quality report

The `mock` provider returns canned turns so everything below runs without an API
key. The demo order tools ship in `sora_agent/tools.py`, registered at import
time by the `@tool` decorator.

```python
from sora_agent import Agent, MockProvider, POStateMachine
from sora_agent.approval import ApprovalGate
from sora_agent.tools import registry


class AutoApprove(ApprovalGate):
    """Approve everything in this demo; production gates ask a human."""
    def request(self, action: str, detail: str) -> bool:
        return self.approve_for(action, detail)


agent = Agent(
    provider=MockProvider(),
    registry=registry.copy(),          # keep the demo's global registry pristine
    approval=AutoApprove(),
    memory=False,                      # keep the sample dependency-free
    identity=False,
    calendar=False,                    # no files written for this demo
)

sm = POStateMachine(po_id="PO-1234")
result = agent.run(
    "Purchase order PO-1234 is stuck; the supplier can't ship on time. "
    "Expedite it.",
    state=sm,
)

print(result.final_answer)   # model's answer
print(result.report)         # EvalReport: [PASS] score=1.00 ...
print(result.stop_reason)    # "completed"
print(sm.state)              # State.EXPEDITED — transitioned only after approval
```

What happens: the model calls `find_tools` to discover `expedite_order` (which is
hidden and marked `needs_approval=True`), the executor pauses for the approval
gate, the tool runs on approval, the state machine transitions, and the quality
gate grades the finished trajectory. Expected output: `state=State.EXPEDITED`,
`stop_reason=completed`, and a `[PASS] score=1.00` report.

### Adding your own tool

```python
from typing import Annotated

from sora_agent.tools import Tool, ToolRegistry


def refund_order(
    order_id: Annotated[str, "Purchase order id, e.g. PO-1234"],
    reason: Annotated[str, "Why the refund is needed"] = "",
) -> str:
    return f"PO {order_id}: refund queued ({reason or 'no reason given'})"


tools = ToolRegistry()
tools.add(Tool(
    refund_order,
    name="refund_order",
    description="Refund a purchase order.",
    needs_approval=True,               # always routed through the approval gate
))
```

Argument descriptions come from `Annotated[str, "..."]`; `Literal`/`Enum`
annotations become JSON-schema enums; optional parameters and scalar defaults
are preserved. See [Tools](#tools).

### Using a real model

```python
from sora_agent.providers import default_provider

provider = default_provider("openai")   # also: azure, deepseek, qwen, mock
```

| Kind | Backend | Required env |
| --- | --- | --- |
| `mock` / `demo` | `MockProvider`, no network | — |
| `openai` / `azure` | OpenAI-compatible chat completions | `OPENAI_API_KEY` (plus optional `OPENAI_MODEL`, default `gpt-5.4-mini`, and `OPENAI_BASE_URL`) |
| `deepseek` / `ds` | DeepSeek | `DEEPSEEK_API_KEY` (optional `DEEPSEEK_MODEL`, default `deepseek-flash`) |
| `qwen` / `dashscope` | DashScope compatible mode | `DASHSCOPE_API_KEY` (optional `QWEN_MODEL`, default `qwen3.7-plus`) |
| `custom` | Any OpenAI-compatible endpoint | `SORA_BASE_URL`; the key is optional and is never borrowed from `OPENAI_API_KEY` |

A real provider with no key raises `ProviderConfigError` naming the exact
environment variable; keys are never reused across providers. Plain HTTP base
URLs are rejected unless they point at loopback.

`azure` is an alias for the same OpenAI-compatible provider; set
`OPENAI_BASE_URL` to your deployment endpoint (and `OPENAI_MODEL` to the
deployment name). The provider is a thin wrapper, so any OpenAI-compatible
gateway works the same way.

### Running the bundled examples

```bash
python examples/po_workflow.py               # interactive: asks y/N per approval
python examples/po_workflow.py --yes         # auto-approve, non-interactive
python examples/po_workflow.py --provider openai
python examples/copilot_function.py          # Azure Function-style entry point
```

Both examples run without `pip install -e .` (they adjust `sys.path`) and use
`examples/_demo_memory.py`, an offline stand-in for the real memory hook.

---

## Architecture

```
              library callers                  chat platforms
                    │                                │
                    ▼                                ▼
               Agent facade                     ChatHandler
                    │                                │
                    └───────────────┬────────────────┘
                                    ▼
                          AgentRuntime.run(RunRequest)
                     (per-run registry copy, tool gate, memory
                      tool, required-first-tool policy, executor)
                                    │
                                    ▼
                                 Runner
                           pure model/tool loop
                        ┌───────────┴───────────┐
                        ▼                       ▼
                     Provider             ToolExecutor
             clock → identity → state →   validate → approve →
             your base provider           execute → truncate → audit
                                                │
                                                ▼
                                           ToolRegistry
```

The layers are intentionally separable:

| Layer | Module | Responsibility |
| --- | --- | --- |
| Agent facade | `sora_agent/agent.py` | Assemble the system prompt, wire memory/identity/calendar/clock, execute the run, evaluate it, and map the result to `AgentResult`. |
| Per-run runtime | `sora_agent/runtime.py` | Build one isolated run: registry copy, `find_tools` gate, memory tool, `ToolExecutor`. |
| Pure loop | `sora_agent/runner.py` | Ask the provider, execute tool calls, feed observations back, stop on answer / budget / policy. Knows nothing about sessions, files, or the CLI. |
| Providers | `sora_agent/providers.py` | `Message`, `ModelTurn`, `ToolCall`, `MockProvider`, `OpenAIChatProvider`, text-`ACTION:` parsing. |
| Tools | `sora_agent/tools.py` | `Tool`, `ToolRegistry`, `ToolExecutor`, `ToolResult`, schema generation, the `@tool` decorator. |
| Attachments | `sora_agent/files.py` | Convert files and images into untrusted markdown carried in a user turn. |
| Untrusted framing | `sora_agent/untrusted.py` | Random per-turn boundaries for attachment and vision text. |
| Discovery | `sora_agent/tool_search.py` | `ToolGate`, `FindToolsTool`, two-tier lexical ranking with aliases and IDF weights. |
| Approval | `sora_agent/approval.py` | `ApprovalGate`, `AuditEntry`; chat variant in `channels/approval.py`. |
| Evaluation | `sora_agent/eval.py` | `Trajectory`, `EvalReport`, `QualityGate`, `structural_grader`, metrics. |
| Judge | `sora_agent/judge.py` | Optional sampled, calibrated, advisory trajectory judge. |
| Channels | `sora_agent/channels/` | Buses, manager, chat handler, sessions, approvals, ten transports, plugin discovery. |
| Memory | `sora_agent/memory/` | Daily verbatim store, CortexDB client/sidecar, recall, dreaming, backups. |
| Model-facing extras | `clock.py`, `calendar.py`, `identity*.py`, `state*.py` | Time, schedule, persona/profile, session state. |
| Ancillary | `isolation.py`, `rag.py` | Process-isolated tool calls; a minimal retriever protocol. |

### `Agent` options

`Agent` is the library entry point; `ChatHandler` takes a similar subset for
channel use.

| Option | Default | Meaning |
| --- | --- | --- |
| `provider` | required | Any `TurnProvider` or `TextProvider`. |
| `registry` | empty `ToolRegistry()` | Tool catalog. `find_tools` is a reserved name. |
| `approval` | `ApprovalGate()` (stdin) | Gate used by tools marked `needs_approval`. |
| `quality` | `QualityGate()` | Post-run structural evaluation. |
| `memory` | on (`default_memory()`) | Memory hook; `False` disables it. |
| `identity` | on (`default_identity()`) | USER.md/SOUL.md injection; `False` disables it. |
| `max_steps` | `10` | Provider turns allowed per run. |
| `max_tool_result_chars` | `4000` | Truncation cap for tool observations. |
| `tool_timeout` | `30.0` | Timeout for `isolated=True` tools (seconds). |
| `provider_retries` | `2` | Retries for transient provider failures. |
| `on_event` | `None` | `RunEvent` observer; it can never affect a run. |
| `clock` | on (`default_clock`) | Current-time prompt section; `False` disables it. |
| `calendar` | on (`default_calendar`) | Calendar store and tools; `False` disables them. |
| `calendar_db` | `~/.sora_agent/calendar.db` | Calendar database path override. |

`memory`, `identity`, `clock`, and `calendar` are resolved at construction and
fail open: a missing optional dependency, corrupt file, or bad configuration
logs a warning and leaves the rest of the agent working.

### `AgentResult`

`agent.run(task, state=None)` returns an `AgentResult`:

| Field | Meaning |
| --- | --- |
| `task` | The task text. |
| `proposal` | Final answer, or the last assistant text if the model never produced one. |
| `final_answer` | The model's final answer (empty if the run did not complete). |
| `stop_reason` | Why the run stopped (see the table under [Runner](#the-modeltool-loop-runner)). |
| `steps` | The `Step` list: one assistant turn plus the tool results it produced. |
| `action` / `tool_result` | The last executed tool call and its (truncated) content. |
| `approved` | Whether any approval-required call was approved. |
| `audit` | `AuditEntry` list from the approval gate. |
| `state` | `state.state` when a state object was passed. |
| `report` | The post-run `EvalReport`. |
| `error` | Provider/runner error text, if any. |

Passing `state=` also enables state transitions on successful tool calls:
`expedite_order → EXPEDITED`, `cancel_order → CANCELLED`,
`defer_order → DEFERRED`, `reroute_order → REROUTED`. A rejected approval never
transitions.

The bundled `POStateMachine` (`sora_agent/state_machine.py`) models:

```
open ──► deferred | expedited | cancelled | rerouted | exception
  any of those ──► executed ──► closed
```

## The model/tool loop (Runner)

`Runner.run(RunSpec(...))` is the pure loop. `Agent`, `ChatHandler`, and any
custom integration all funnel through it via `AgentRuntime`.

### `RunSpec` (inputs)

| Field | Default | Meaning |
| --- | --- | --- |
| `messages` | required | Conversation so far, including the system prompt. |
| `tools` | required | Initial visible tools. |
| `max_steps` | `10` | Provider turns allowed. `Agent` default 10; `ChatHandler` default 8. |
| `tool_provider` | `None` | Recompute the visible tool list before every step (lazy discovery). |
| `should_stop` | `None` | Cooperative cancellation, checked between steps and calls. |
| `provider_retries` | `2` | Retries for transient provider failures. |
| `retry_base_delay` / `retry_max_delay` | `0.5` / `8.0` | Exponential backoff window, equal jitter. |
| `required_first_tool` | `None` | The model's first action must be this tool. |
| `on_event` | `None` | `RunEvent` observer; an observer can never affect a run. |

### `RunResult` (outputs)

`RunResult` carries the full message transcript (with tool observations
appended), `steps`, `final_answer`, `stop_reason`, `error`, `metrics`
(`RunMetrics`), `run_id`, `report`, plus `tool_schemas` /
`tool_requirements` snapshots used by evaluation.

### Stop reasons

| `stop_reason` | Trigger |
| --- | --- |
| `completed` | The model answered without calling a tool. |
| `max_steps` | Step budget exhausted. |
| `cancelled` | `should_stop()` returned true. |
| `provider_error` | Provider raised and retries were exhausted. |
| `tool_error` | The tool executor itself crashed. |
| `approval_rejected` | A human denied the action, or the gate failed/timed out. |
| `<tool>_required` | The required first tool (e.g. `memory_recall_required`) was skipped repeatedly; the third violation hard-stops the run. |
| Any tool's `stop_reason` | A tool may end a run deliberately (e.g. a policy stop). |

### Reliability behavior

- **Retries**: transient provider failures (timeouts, connection errors, 408 /
  409 / 425 / 429, and 5xx) retry with exponential backoff and equal jitter.
  Authentication, permission, validation, and configuration errors fail
  immediately instead of burning the retry budget. Backoff sleeps in 50 ms
  slices so cancellation is immediate.
- **Required first tool**: when set (or when memory is enabled — see
  [Memory](#memory-cortexdb)), a direct answer or wrong tool is rejected with a
  correction message. The first two violations get a correction; the third
  hard-stops with `<tool>_required`. Skipped calls are recorded in the transcript
  so the assistant/tool message pairing stays valid.
- **Many calls per turn**: all native tool calls in a turn execute in order;
  after a *required first tool* completes, remaining calls from that same turn
  are skipped so the model can see the result before acting on it.
- **Truncation**: tool output is truncated to `max_tool_result_chars`
  (`Agent`/`ChatHandler` default 4000) with an explicit
  `...[truncated N chars]` marker.
- **Redundancy telemetry**: identical `name + sorted-args` calls are counted in
  `metrics.redundant_tool_calls` and penalize the quality score, but are not
  blocked.
- **Events**: `run_started`, `provider_finished`, `provider_retry`,
  `tool_finished`, `run_finished` — each with step, name, duration, ok, detail.
- **Metrics**: durations, provider calls/retries, tool calls/errors/executions/
  skips, approval denials, redundant calls, isolated calls, and token usage when
  the provider reports it.

## Providers

A provider is anything matching one of two protocols:

```python
class TurnProvider(Protocol):
    def complete_turn(self, messages: list[Message], tools: list[Tool]) -> ModelTurn: ...

class TextProvider(Protocol):
    def complete(self, messages: list[Message], tools: list[Tool]) -> str: ...
```

`ModelTurn` carries `text`, `tool_calls: list[ToolCall]`, and optional
`TokenUsage`. Providers return one completed turn; chat channels show a typing
indicator while the turn is running.

Text-only providers may speak a legacy single-action protocol — exactly one
line such as:

```
ACTION: expedite_order(order_id="PO-1234")
```

Parsing happens once, at the provider edge, so the runner only ever sees
structured turns. Providers that only implement `complete` automatically get
this behavior.

Provider wrappers compose transparently (each guards against double-wrapping).
From the outermost call inwards:

```
ClockProvider (applied last, current time)
    └─► IdentityProvider (USER/SOUL)
            └─► StatefulProvider (optional, session state)
                    └─► your base provider
```

`MockProvider` follows the production path offline: recall first when memory is
visible, discover a hidden business tool with `find_tools`, then call it.
`OpenAIChatProvider` is a thin Chat Completions wrapper with native tool calls
and usage reporting.

## Tools

### Defining tools

```python
# Decorator form — registers into sora_agent.tools.registry
from sora_agent.tools import tool

@tool("lookup_order", "Return a purchase order's status and details.")
def lookup_order(order_id: str) -> str:
    return f"PO {order_id}: status=open"

# Constructor form — full control
from sora_agent.tools import Tool, ToolRegistry


def refund_order(order_id: str) -> str:
    return f"PO {order_id}: refund queued"


registry = ToolRegistry()
registry.add(Tool(
    refund_order,
    name="refund_order",
    description="Refund a purchase order.",
    needs_approval=True,
))
```

`Tool` options:

| Option | Meaning |
| --- | --- |
| `needs_approval=True` | Routed through the approval gate before execution; the schema description gets a `[approval required]` marker. |
| `isolated=True` | Runs in a spawned child process so a hang can actually be killed (`tool_timeout`, default 30 s). Arguments and return value must be picklable. The result pipe is read while the child runs, so a large result cannot cause a false timeout. This is crash/hang isolation, **not** a security sandbox. |
| `requires=("lookup_order",)` | Declares prerequisites; the structural grader checks they succeeded earlier. |
| `aliases=("meeting", "appointment")` | Extra trigger words used by lexical tool search. |

### Schema generation

`Tool.schema` is derived from the function signature and type hints:

- `Annotated[str, "description"]` adds a parameter description.
- `Literal[...]` and `Enum` classes become JSON-schema `enum`s.
- `Optional[T]` / `T | None` unwrap to `T`.
- Scalar defaults (`str`, `int`, `float`, `bool`) are emitted as `default`.
- `*args` / `**kwargs` are ignored; parameters without defaults are required.
- The result is JSON-serializable and ready for OpenAI-style function calling.

### Execution pipeline

`ToolExecutor.execute(call)` is the only place a model tool call becomes a
result, in this order:

1. Unknown tool → error observation.
2. Malformed JSON arguments (`call.error`) → error observation.
3. Arguments must be an object and must bind against the callable's signature.
4. `needs_approval` → `ApprovalGate.request(...)`. Any gate error, missing gate,
   or denial **fails closed** with `stop_reason="approval_rejected"`.
5. Execute directly, or via `run_isolated` for `isolated=True`.
6. Tool exceptions become observations (`tool call failed: ...`) — they never
   crash the loop.
7. Truncate, run the `after_execute` policy hook, and log to the audit trail.

`ToolResult` fields: `call`, `content`, `is_error`, `approval_required`,
`approved`, `executed`, `stopped`, `stop_reason`, `report`, `skipped`,
`isolated`.

## Lazy tool discovery

The initial tool list is intentionally partial. At the start of a run the model
sees only:

- `find_tools` — always;
- `memory_recall` — when memory is enabled (and it must be the first action).

Everything else stays registered but hidden. The hidden catalog is rendered
*inside the `find_tools` schema* (name + one-line purpose, capped at 40 entries)
so the prompt prefix stays small and stable.

`find_tools(need="...")` ranks registered tools lexically — token overlap with
IDF weights, optional tool aliases, plus `difflib.SequenceMatcher` for typos —
and returns a preview of up to 8 matches (minimum score 2.0) **without loading
them**. When one match is clearly dominant, it loads automatically so the common
case stays one step.

`find_tools(load=["name", ...])` then loads exact names (case-insensitive and
space/underscore tolerant), at most 3 per call. `ToolGate` also caps the run at
12 visible tools, so repeated searches cannot balloon the prompt. Unlocked tools
appear in the model's tool list on the next step. Discovery is per-run:
`AgentRuntime` copies the registry and builds a fresh `ToolGate`, so one run's
unlocks never leak into another. `find_tools` is a reserved name — registering a
tool with it raises `ValueError`.

The ranking strategy is behind a `ToolRanker` protocol (`LexicalRanker` today),
so an embedding backend can replace it without changing the `find_tools`
contract.

## Human approval

`ApprovalGate` is synchronous and blocking:

```python
from sora_agent.approval import ApprovalGate

gate = ApprovalGate()
approved = gate.request("expedite_order", 'ACTION: expedite_order(order_id="PO-1234")')
# prompts on stdin: "Approve? [y/N]"
```

Every decision is appended to `gate.audit` as an `AuditEntry`
(`ts`, `action`, `actor`, `decision`, `detail`). For tests and non-interactive
runs use `gate.approve_for(action, detail, actor="human-reviewer")`.

For chat channels, `channels/approval.py` provides:

- **`ApprovalBroker`** — publishes the prompt to the chat, blocks the worker
  thread on a `threading.Event`, and resolves it when the reply arrives.
- **`ChannelApprovalGate`** — an `ApprovalGate` whose question goes to the chat
  and whose audit records the sender as the actor.

Behavior details:

- Clear replies (`yes`, `y`, `ok`, `approve`, ... / `no`, `n`, `reject`, ...)
  decide the request; anything else is left for the agent, so a user can ask a
  follow-up question while an approval is pending.
- Only the sender who triggered the turn can decide it. Replies and button
  presses from another participant in the same chat are ignored.
- A timeout (default 300 s in the gateway) **denies** the request.
- Button presses carry an `approval_id`; a button from an older prompt cannot
  decide a newer request. The token also resolves topic prompts whose callback
  arrives without a session key.
- Approval prompts keep the Telegram forum topic (`message_thread_id`). The
  button spinner is acknowledged only after the broker consumes the callback;
  stale buttons answer "This approval is no longer active" and keep their
  keyboard instead of claiming approval.
- `/reset` cancels a turn that is still running and bumps the session
  generation, so a late reply cannot resurrect the old history. `/cancel`
  releases a worker blocked on an approval immediately.
- `/cancel` and gateway shutdown release blocked approvals immediately.

## Quality gate and evaluation

Evaluation is deliberately **non-executional**: no tool call may depend on a
verdict. Safety lives in the approval gate and the tool policy. A grader error
therefore produces a report, not a block.

```python
from sora_agent.eval import QualityGate, trajectory_from_run

# Agent evaluates automatically:
result = agent.run(task)
print(result.report)   # [PASS] score=1.00 / [FAIL] / [SKIP]

# Or grade a finished Runner run yourself (`run` is a RunResult):
report = QualityGate().evaluate(
    trajectory_from_run(task, run, final_answer=final_text)
)
print(report.passed, report.score, report.checks, report.dimensions)
```

`structural_grader` (the default) checks what a finished trajectory can prove:

- the run stopped with `stop_reason="completed"` (denials, cancellation,
  provider errors, and budget exhaustion fail even if text was produced);
- the final answer is non-empty;
- every called tool is in the tool set;
- arguments conform to the tool's JSON schema (required fields and types);
- declared `Tool.requires` prerequisites succeeded earlier;
- a tool never executes after the run stopped;
- a tool never executes without the approval it required;
- the required first tool (when set) was actually first;
- skipped calls are reported, and repeated identical calls are reported (score
  penalty, not a hard failure).

It deliberately does **not** check semantic correctness, grounding of claims in
tool results, safety/desirability of side effects, business rules, idempotency,
prompt injection, whether a better tool existed, undeclared dependencies, or
argument semantics beyond schema shape.

`trajectory_metrics(trajectory)` derives efficiency measures: action success
rate (successful executions divided by attempted calls, including skipped
calls), redundant-call rate, steps to answer, attempted/successful calls, token
counts, and provider/tool seconds.

### Chat inspection and control

`ChatHandler` exposes the latest report and cooperative control over chat:

- `/status` — last run's stop reason, steps, tool calls, errors, duration,
  tokens, action-success rate, redundancy, and eval verdict.
- `/eval` — the full latest `EvalReport`.
- `/cancel` — cooperatively stop the running turn.
- `/reset` — forget the conversation (in memory **and** on disk).

## Optional trajectory judge

`judge.py` adds a sampled LLM judge. It is **off by default** and never
participates in execution: its verdict cannot approve, block, or alter a tool
call. One judged run costs one provider call, so sampling is the budget control.

```python
from sora_agent.judge import JudgeConfig, JudgeGrader

judge = JudgeGrader(
    provider=judge_provider,                      # a different model family
    config=JudgeConfig(
        sample_rate=0.1,
        calibration_path="judge_calibration.json",
        agent_family="gpt-4o-mini",
        model_family="claude-sonnet",
    ),
)
agent.quality = QualityGate(grader=judge)
```

Guardrails:

- **Calibration gate**: `JudgeGrader` refuses to score until a calibration
  manifest meets minimum agreement (0.80), Cohen's kappa (0.60), and recall on
  bad trajectories (0.80). Calibration values must be finite and in range; a
  NaN or infinite metric rejects the manifest instead of passing every
  comparison. Non-finite dimension scores in a judge response are discarded.
- **Family rule**: `agent_family` and `model_family` are declarations, not
  verifiable facts, but using the same family is rejected.
- **Advisory safety**: the `safety` dimension appears in the report and lowers
  the score, but never decides `passed` and never influences execution.
- `calibrate_judge(judge, labelled)` and `calibration_metrics(labels, preds)`
  compute agreement/kappa/recall against human labels; `save_calibration` /
  `load_calibration` persist the manifest.

## Clock

The loop has no inherent sense of "now". `ClockProvider` appends one small
per-turn system section so the model can resolve "today", "tomorrow", and
relative deadlines:

```
## CURRENT TIME
Local: 2026-11-04T18:01:23+08:00 (Wednesday, CST)
UTC: 2026-11-04T10:01:23Z
Treat this as the authoritative current time. ...
```

- Enabled by default for `Agent` and `ChatHandler`; disable with `clock=False`.
- The section is always ordered last: after the base prompt, identity, and
  state. The identity/state wrappers use `extract_clock`/`append_clock` to
  preserve that ordering no matter which order wrappers are composed in.
- `resolve_clock` accepts `False`, `True`/`None`/`"default"`, a fixed
  `datetime` (useful in tests), or any callable returning a datetime. A broken
  clock fails open and never breaks a turn.

## Calendar

A small, dependency-free calendar the model reaches through ordinary tools.

- Storage: one SQLite file at `~/.sora_agent/calendar.db`
  (override with `SORA_CALENDAR_DB`).
- Item kinds: `event` (title, start, optional end) and `todo` (title, optional
  due date, completed flag, completion timestamp).
- Importance slider with exactly three stops: `important`, `balanced`,
  `not important` (stored with a sort rank; common aliases like "high" or "low"
  are normalized).
- Dates accept ISO 8601 (`2026-11-05`, `2026-11-05 14:30`,
  `2026-11-05T14:30:00+08:00`) plus `now`, `today`, `tomorrow`, `yesterday`.
  Naive values use the clock's timezone.

Tools registered on the (hidden) registry — reached via `find_tools`:

| Tool | Purpose |
| --- | --- |
| `calendar_list` | List events/to-dos in a window (default: today onward, 30 days; to-dos without due dates included). |
| `calendar_add_event` | Add an event with title, start, optional end, notes, importance. |
| `calendar_add_todo` | Add a to-do with optional due date, notes, importance. |
| `calendar_complete_todo` | Mark a to-do done. |
| `calendar_set_importance` | Move an item's importance slider. |

`Agent(calendar=None)` and `ChatHandler()` register the calendar by default;
disable with `calendar=False` or point at another file with
`calendar_db="/path/to.db"` (also `--no-calendar` / `--calendar-db` on the
gateway).

## Session state

`state.py` is a self-contained, session-scoped persistent state document — the
agent's cumulative working memory for one conversation. It uses only the
standard library and knows nothing about the loop; `state_provider.py` is the
single adapter.

- **Storage**: `<session_id>.state.md` inside a caller-supplied root (there is
  deliberately no default root). Writes are atomic (`fsync` + `os.replace`,
  0600 files, 0700 root); the previous valid document is kept as
  `<session_id>.state.md.bak`.
- **Format**: exactly four `## ` headings in order —

  ```
  ## Goal
  ...
  ## Current State
  ...
  ## Next to do if prompted or approved to
  ...
  ## User
  Feeling: ...
  Thinking: ...
  ```

- **Validation** (`StateMonitor`, pure and deterministic): missing/duplicate/
  misplaced headings, empty sections, placeholders like `TODO`, code fences,
  control/invisible characters, reserved `<state_edit>` sentinels, gibberish
  (repeated tokens, long runs, implausible tokens), clearing a previously
  non-empty section or a `Feeling:`/`Thinking:` value, and character limits.
- **Limits**: length 2048–8096 chars (default 4076); repair attempts 1–5
  (default 3). State is **never** deleted, cleared, or reset. A corrupt file is
  never overwritten by the recovery path; rejected edits leave the file
  untouched; an unchanged edit is a no-op.
- **Protocol**: the provider wrapper appends a state protocol plus the current
  document inside `<<<STATE ... STATE>>>`; the model may append at most one
  full-replacement `<state_edit>...</state_edit>` block. Invalid edits trigger a
  bounded deterministic repair loop, and the visible answer is never rewritten
  by a repair.
- **Prompt caching**: the original system messages stay byte-for-byte untouched;
  the state is inserted as the final system message (before the clock).

```python
from pathlib import Path

from sora_agent.agent import Agent
from sora_agent.providers import MockProvider
from sora_agent.state import StateConfig, StateManager
from sora_agent.state_provider import StatefulProvider

manager = StateManager(
    "po-1234",
    StateConfig(root=Path.home() / ".sora_agent" / "state"),
)
manager.ensure("Handle the stuck PO exception")   # creates the first document

agent = Agent(
    provider=StatefulProvider(MockProvider(), manager),
    memory=False,
    identity=False,
    calendar=False,
)
```

## Identity (USER.md / SOUL.md)

Persistent, file-backed identity for long-lived assistants.

- `USER.md` — the single authoritative place for durable facts and preferences
  about the user (limit 1000 chars).
- `SOUL.md` — the single authoritative place for the agent's stable persona,
  voice, values, and boundaries (limit 1500 chars).
- There is deliberately **no** `AGENTS.md` support and no directory scan; only
  exactly these two documents are ever loaded.
- Default location `~/.sora_agent/identity` (override `SORA_IDENTITY_ROOT`,
  disable with `SORA_IDENTITY=0`). For `Agent`, identity is **on by default**
  and opt-out; the gateway enables it with `--identity`.

Document format is intentionally strict — one fact per line:

```
- Name: Jaden
- Age: 24
- Likes: potatoes
```

Validation rejects headings, prose, code fences, placeholders, duplicate keys,
keys longer than 40 chars, values longer than 200 chars, values that look like
secrets (keys containing "password"/"token"/"api key"/..., or values matching
`sk-...`, GitHub tokens, AWS keys, PEM headers), reserved edit markers, and
documents over the limit.

The model updates identity with exactly one full-replacement block per
document, `<user_edit>...</user_edit>` and/or `<soul_edit>...</soul_edit>`.
Blocks are removed from the visible answer before the user sees it; rejected
edits go through a bounded repair loop that never rewrites the answer.

Durability: every accepted edit archives the previous active backup and writes
new gzip backups with a `manifest.json` carrying compressed/plaintext SHA-256
hashes and character counts. Backups are verified after write; corrupt primary
files fall back to the active backup and are never overwritten; there is an OS
file lock per document; retention defaults to 3 payloads per document.

The **memory firewall** lives here: `IdentityStore.redact_for_memory(text)`
removes known profile terms/bullets and (for user messages) common explicit
profile/PII patterns before anything is written to memory or returned from
recall, replacing them with `[profile]` / `[profile details withheld]`.
Credentials are removed for every source, including assistant output: OpenAI
and Stripe keys, GitHub tokens, AWS and Google keys, Slack tokens, bearer
tokens, JWTs, PEM private keys, and `password: ...` / `token = ...`
assignments become `[secret]`.

| Env var | Default | Meaning |
| --- | --- | --- |
| `SORA_IDENTITY` | `true` | Master switch (set `0`/`false` to disable). |
| `SORA_IDENTITY_ROOT` | `~/.sora_agent/identity` | Storage root. |
| `SORA_IDENTITY_RETENTION` | `3` (1–20) | Backup payloads kept per document. |
| `SORA_IDENTITY_MAX_REPAIRS` | `2` (0–5) | Repair attempts for invalid edits. |

## Memory (CortexDB)

An optional long-term memory subsystem (`sora_agent/memory/`). The core modules
are stdlib-only and know nothing about the agent loop; `memory/hook.py` is the
single adapter.

### How it works

```
Agent.run/ChatHandler turn
   │  memory_recall (required first action, always visible)
   ▼
MemoryRecallService ──► DailyStore (today/yesterday, verbatim SQLite)
   │                └─► CortexDB sidecar (long-term, JSON over loopback)
   ▼
after_run ──► redact via identity firewall ──► DailyStore append
   │
   ▼ nightly (DreamScheduler) or `sora-memory dream`
DreamRunner: backup → transfer daily→CortexDB → dedupe → conflicts →
graph hints → Louvain communities → inferred links → optional rules →
decay → expire → mark → sweep expired → report → artifact cleanup
```

- **Daily verbatim memory**: one SQLite file per UTC day
  (`~/.sora_agent/memory/daily/YYYY-MM-DD.db`, WAL mode, optional FTS5 with a
  `LIKE` fallback). Only user and assistant messages are written — never tool
  output or system prompts. Legacy files without an FTS index are rebuilt once
  on open (tracked with `PRAGMA user_version`), and search tokens are Unicode,
  with non-ASCII queries using substring matching so CJK and accented text work.
- **Recall-first policy**: when memory is enabled, `memory_recall` is injected
  as always-visible and set as the required first tool; the system prompt tells
  the model to recall before acting. Results are framed as *untrusted data, not
  instructions*. Scopes: `auto` (daily + long-term), `daily`, `longterm`.
  Every hit passes back through the identity redactor.
- **Long-term store**: an external `cortexdb-grpc` sidecar that owns the active
  database. Python speaks JSON over loopback (no gRPC/protobuf dependency) on a
  dynamically chosen free port with a bearer token at
  `~/.sora_agent/memory/grpc.token`.
- **Sidecar lifecycle**: `SidecarManager` downloads a platform binary for
  `SORA_MEMORY_SIDECAR_VERSION` (default `2.139.1`) from
  `SORA_MEMORY_DOWNLOAD_BASE_URL` (default: the `liliang-cn/cortexdb` GitHub
  releases), verifies the artifact against a pinned checksum, extracts it to
  `~/.sora_agent/bin/cortexdb-grpc`, starts it with an allowlisted environment,
  and polls `/v1/health` until it is up (default 20 s). Downloads refuse
  plaintext HTTP to non-loopback hosts (including redirects) and each install
  is protected by a single-instance lock. Existing binaries must match the
  checksum recorded at download time (or `SORA_MEMORY_SIDECAR_SHA256`) before
  they are executed. A dead client is probed and restarted every few seconds
  instead of being cached forever. `sora-memory download|start|stop|doctor`
  exposes the lifecycle.
- **Fail-open**: if the sidecar is unavailable, long-term recall degrades to
  daily results with an explicit note and the backend is retried at most every
  30 seconds. A broken memory hook never breaks a run.
- **Dream consolidation**: pending day files (strictly before today) are
  verified-saved to CortexDB with deterministic ids
  (`daily:<date>:<row id>`), then only the transferred rows are deleted. Rows
  appended while the transfer is running stay in the day file for the next
  run, and a prior partial run is skipped by id instead of duplicated. The
  dream pipeline deduplicates near-identical memories, flags conflicting
  numeric claims (which can engage the optional wake model), builds a
  token-overlap graph, runs a dependency-free deterministic Louvain community
  pass, infers two-hop links, optionally applies configured rules, decays
  importance (default ×0.98, floor 0.05), expires records, and writes a JSON
  report to `~/.sora_agent/memory/dream/reports/<run_id>.json`. Dream artifacts
  (merged records, conflicts, inferred links) are transient and cleaned up at
  the end of the run. A file lock (`dream.lock`) makes concurrent dreams fail
  fast with `DreamBusy`.
- **Scheduling**: when memory is enabled and `SORA_MEMORY_DREAM_AUTO` is true, a
  daemon thread (`sora-memory-dream`) runs one dream at the configured UTC hour
  (default midnight) and performs a startup catch-up if work is pending.
- **Backups**: `VACUUM INTO` + `PRAGMA integrity_check`, zstd compression,
  SHA-256 manifests, and retention of 4 active / 14 daily / 8 weekly / 12
  monthly snapshots. Missing zstd support is reported as an actionable
  `BackupError`. Pruning accepts only plain filenames that resolve inside the
  backup directory, so a corrupt or malicious manifest cannot delete files
  elsewhere (including through symlinks). Restore verifies the manifest checksum
  and integrity, stops the sidecar, preserves the previous DB as
  `*.pre-restore-<stamp>`, and restarts the sidecar.

### Key configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `SORA_MEMORY` | `true` | Master switch (opt-out). |
| `SORA_MEMORY_ROOT` | `~/.sora_agent` | Root for db, daily files, backups, logs. |
| `SORA_MEMORY_USER_ID` | `sora` | Scope used when writing/recalling memories. |
| `SORA_MEMORY_RECALL_TOP_K` | `8` (1–50) | Recall result budget. |
| `SORA_MEMORY_RECALL_MAX_CHARS` | `6000` | Rendered recall text cap. |
| `SORA_MEMORY_RECALL_MODE` | `auto` | Retrieval mode: `auto`, `lexical`, `graph`, `ppr`. |
| `SORA_MEMORY_AUTO_DOWNLOAD` | `true` | Download the sidecar when missing. |
| `SORA_MEMORY_DOWNLOAD_BASE_URL` | CortexDB release base URL | Where to download the sidecar from. |
| `SORA_MEMORY_SIDECAR_BIN` | — | Use an existing binary instead. |
| `SORA_MEMORY_SIDECAR_VERSION` | `2.139.1` | Release to download. |
| `SORA_MEMORY_SIDECAR_SHA256` | — | Pin the expected binary checksum (64 hex characters). |
| `SORA_MEMORY_HTTP_ADDR` / `SORA_MEMORY_GRPC_ADDR` | `127.0.0.1:0` | Listen addresses (port 0 = free port). |
| `SORA_MEMORY_TOKEN` | generated file | Bearer token for the HTTP API. |
| `SORA_MEMORY_REQUEST_TIMEOUT` / `SORA_MEMORY_START_TIMEOUT` | `10.0` / `20.0` | Seconds. |
| `SORA_MEMORY_DREAM_AUTO` | `true` | Automatic nightly dreams. |
| `SORA_MEMORY_DREAM_HOUR_UTC` | `0` | Hour of the nightly run. |
| `SORA_MEMORY_DREAM_ROUNDS` | `2` | Consolidation rounds per memory. |
| `SORA_MEMORY_DREAM_BUDGET` | `120` | Pipeline seconds budget. |
| `SORA_MEMORY_DREAM_EDGE` | `0.45` | Jaccard threshold for graph edges. |
| `SORA_MEMORY_DREAM_DECAY` / `SORA_MEMORY_DREAM_MIN_IMPORTANCE` | `0.98` / `0.05` | Importance decay and floor. |
| `SORA_MEMORY_BACKUP_ACTIVE` / `_DAILY` / `_WEEKLY` / `_MONTHLY` | `4` / `14` / `8` / `12` | Retention counts. |
| `SORA_MEMORY_WAKE_MODEL`, `SORA_MEMORY_WAKE_BASE_URL`, `SORA_MEMORY_WAKE_API_KEY`, `SORA_MEMORY_WAKE_TIMEOUT` | — | Optional wake-model review of conflict-bearing dream reports. |

Additional caps: `SORA_MEMORY_DREAM_MAX_MEMORIES` (500),
`SORA_MEMORY_DREAM_MAX_MERGES` (20), `SORA_MEMORY_DREAM_MAX_CONFLICTS` (20),
`SORA_MEMORY_DREAM_MAX_INFERENCES` (20), `SORA_MEMORY_DREAM_MAX_UPDATES` (300),
and `SORA_MEMORY_RULES` (a JSON array of rule dicts).

### Using memory

```python
from sora_agent import Agent
from sora_agent.memory import default_memory

agent = Agent(provider=provider)              # memory on by default (opt-out)
# or explicitly:
agent = Agent(provider=provider, memory=default_memory())
# or disable:
agent = Agent(provider=provider, memory=False)
```

After each run the hook records the (redacted) user message and the assistant's
reasoning/final text to daily memory. Tool output and system prompts are never
stored. Identity edits apply *before* the user message is buffered, so the
firewall always redacts against the updated profile.

## Chat channels and the gateway

`sora_agent/channels/` decouples transports from the agent:

| Piece | Role |
| --- | --- |
| `InboundMessage` / `OutboundMessage` | Channel-agnostic payloads (channel, sender, chat, content, media `file_id`s, metadata). |
| `MessageBus` | Two asyncio queues (inbound/outbound) so a slow platform never blocks the model loop. |
| `BaseChannel` | Transport ABC: `start()` (blocks), `stop()` (idempotent), `send()`, an optional typing indicator, plus the shared `allow_from` authorization rule (`"*"` allows everyone; an empty list denies everyone — a token is not an access policy). |
| `ChannelManager` | Supervises channels, dispatches commands, runs turns in tasks with per-session locks, bounds the pending-turn queue (default 32) and rate-limits each session (default 20 turns/minute), answers control commands, delivers outbound with retries. If a channel dies, the gateway shuts down rather than half-serving. |
| `ChatHandler` | Per-session history, run assembly, cancellation, `/status`, `/eval`, `/reset`, `/cancel`. |
| `SessionStore` | Durable JSON conversation history, one hashed file per session under `~/.sora_agent/sessions` (override `SORA_SESSIONS_ROOT`). History is trimmed to `max_turns * 2` messages (default 40 turns). |
| `ApprovalBroker` / `ChannelApprovalGate` | Chat-based approvals (see [Human approval](#human-approval)). |

### Built-in channels

Ten transports ship in-box. They all send one final reply after the turn
completes (no token streaming), and each one shows a "preparing" signal while
the agent is working.

| Channel | Inbound | Outbound | Preparing signal | Minimum setup |
| --- | --- | --- | --- | --- |
| Telegram | Bot API long poll | Bot API | `sendChatAction: typing` | Bot token |
| Discord | Gateway WebSocket | REST | `POST /channels/{id}/typing` | Bot token + Message Content intent |
| Slack | Socket Mode | Web API | Assistant status, then a temporary placeholder message | Bot token + app token (`connections:write`) |
| WhatsApp | Cloud API webhook | Graph API | Read receipt with `typing_indicator` | Phone number id, permanent token, verify token, app secret |
| Microsoft Teams | Bot Framework webhook | Bot Framework REST | `typing` activity | Azure Bot app id + client secret |
| Signal | `signal-cli` SSE | `signal-cli` JSON-RPC | `sendTyping` | Running `signal-cli daemon --http` |
| Google Chat | Signed HTTP events | Incoming webhook | Immediate webhook acknowledgement | App project number + space webhook URL |
| Matrix | Client-Server `/sync` | Client-Server API | `m.typing` | Homeserver URL + access token (unencrypted rooms) |
| Feishu / Lark | Event v2 webhook | Open API | `OnIt` reaction, removed after the reply | App id, secret, verification token |
| iMessage | BlueBubbles webhook | BlueBubbles REST | Typing endpoint | BlueBubbles server URL + password |

All channels enforce the same `allow_from` rule: `"*"` allows everyone and an
empty list denies everyone. A token alone is never an access policy.

Select one channel per gateway process with `--channel <name>` or the saved
`channel.kind` setting. Each channel reads its own `SORA_<CHANNEL>_*` variables;
run `sora-gateway --list` to see the installed channel packages.

```bash
export SORA_DISCORD_TOKEN=...
export SORA_DISCORD_ALLOW_FROM=123456789012345678
sora-gateway --channel discord
```

**Platform notes.** Discord needs the Message Content intent enabled for the
bot. Slack uses Socket Mode, so no public URL is required. WhatsApp is the
Cloud API (not WhatsApp Web); Meta must be able to reach the webhook endpoint.
Teams and Google Chat verify RS256 webhook tokens against the platform's JWKS.
Matrix supports unencrypted rooms only. Signal and iMessage depend on their
external daemons being reachable. Webhook channels bind to `127.0.0.1` by
default; set `SORA_<CHANNEL>_WEBHOOK_HOST=0.0.0.0` only behind a TLS-terminating
reverse proxy or tunnel, and keep signature/token verification enabled. Google
Chat has no typing API, so the immediate acknowledgement is the preparing
signal and the final reply is posted through the configured incoming webhook.

### Telegram

The in-box channel long-polls the Bot API using only `urllib` (`getUpdates`),
downloads attachments when file understanding is enabled, and delivers replies in
platform-sized chunks that prefer natural breaks.

- **Typing indicator**: the gateway sends Telegram `sendChatAction` while a
  turn is running, refreshing it periodically and cancelling it on completion.
  Replies are sent once, after the turn completes.
- **Approvals**: approval prompts get Approve/Reject inline buttons
  (`approval:<yes|no>:<id>`); prompts stay inside their forum topic. A button
  press is acknowledged only after the broker consumes it; stale buttons say
  the approval is no longer active and keep their keyboard. Replies by text
  (`yes` / `no`) work too, but only from the sender who triggered the turn.
- **Topics**: a `message_thread_id` scopes the session to
  `telegram:<chat>:<thread>`.
- **Config**: `SORA_TELEGRAM_TOKEN` (required), `SORA_TELEGRAM_ALLOW_FROM`
  (comma-separated user ids, or `*`), `SORA_TELEGRAM_API_BASE`,
  `SORA_TELEGRAM_POLL_TIMEOUT`, `SORA_TELEGRAM_REQUEST_TIMEOUT`,
  `SORA_TELEGRAM_PARSE_MODE` (`HTML`/`Markdown`/`MarkdownV2`),
  `SORA_TELEGRAM_MAX_CHARS` (default 4096),
  `SORA_TELEGRAM_DOWNLOAD_FILES` (default on), and
  `SORA_TELEGRAM_MAX_ATTACHMENT_BYTES` (default 20 MiB).

Start it with:

```bash
export SORA_TELEGRAM_TOKEN=123456:ABC...
export SORA_TELEGRAM_ALLOW_FROM=98765432   # your numeric user id

sora-gateway --provider mock --tools demo --memory --identity
```

### Running the gateway in the background

`sora gateway start` launches the gateway in a new session, writes
`~/.sora_agent/gateway.pid`, and redirects output to
`~/.sora_agent/gateway.log`. Because it is detached from the terminal, it
keeps running after the SSH connection closes:

```bash
sora gateway start
sora gateway status
sora gateway logs -n 100
sora gateway restart
sora gateway stop
```

| Command | Behavior |
| --- | --- |
| `sora gateway start` | Start detached in the background; idempotent if already running. |
| `sora gateway run` / `sora gateway` | Stay in the foreground (the previous behavior). |
| `sora gateway stop` | Send `SIGTERM`, wait up to 15 s, then force-stop if needed. |
| `sora gateway status` | Exit `0` when running, `1` when stopped. |
| `sora gateway restart` | Stop (if running) and start again. |
| `sora gateway logs` | Print the last 50 lines; use `-n` for another count. |

Save the Telegram token with `sora set-key telegram` (or `sora setup`) so the
background process can find it on future starts. Other channels read their
credentials from the `SORA_<CHANNEL>_*` environment variables above.
`SORA_GATEWAY_PID` and `SORA_GATEWAY_LOG` override the PID and log paths.

Background mode survives SSH disconnects but not a reboot. For boot
persistence, run `sora-gateway` under systemd, launchd, or another service
manager; it stays in the foreground and handles `SIGTERM` gracefully.

### Files and images

The **file understander** (`sora_agent/files.py`) downloads and converts
attachments from channels that support them (up to the channel's limit) into
detailed markdown, then sends that markdown as a **user turn wrapped in an
untrusted boundary** before the actual request. It never enters the system
prompt. The latest attachment context stays available in memory for follow-up
turns and is cleared by `/reset`. With a vision model (auto-selected or
configured) or OCR, the main model does not need to support images or file
inputs.

Supported conversions:

| Input | Output |
| --- | --- |
| Text, Markdown, source code, JSON, XML, YAML, TOML, HTML | Fenced text or a cleaned text block; JSON is pretty-printed and HTML scripts/styles are dropped. |
| CSV | Markdown table for small files, fenced CSV otherwise. |
| PDF | Extracted text with the optional `files` extra (`pypdf`). |
| DOCX, PPTX, XLSX | Paragraphs, slide text, and sheet tables using the standard library. |
| ZIP, TAR | Entry listing with sizes; nested archives are not extracted. |
| Images | A configured vision model, then OCR if available, then dimensions/hash metadata. |
| Audio, video, voice notes | Duration/dimensions/title metadata; transcription is not enabled. |
| Other binary | MIME type, size, SHA-256, and a short hex preview. |

Configuration lives under `features.files`:

```bash
sora config set features.files.enabled true
sora config set features.files.max_file_bytes 20971520
sora config set features.files.max_chars 20000
sora config set features.files.total_max_chars 40000
sora config set features.files.vision_model gpt-5.4-mini
```

Environment overrides: `SORA_FILES`, `SORA_FILES_MAX_BYTES`,
`SORA_FILES_MAX_CHARS`, and `SORA_FILES_TOTAL_MAX_CHARS`. Per-channel download
limits are controlled by `SORA_<CHANNEL>_DOWNLOAD_FILES` and
`SORA_<CHANNEL>_MAX_ATTACHMENT_BYTES` (Telegram defaults to 20 MB, the Bot API
limit).

For images, SoraAgent uses the main model automatically when it is known to
accept image input (OpenAI GPT-4o/4.1/5/6 and o3/o4-mini; Qwen3.7 Plus,
Qwen3.8 Max, and Qwen3.8 Omni Flash). Otherwise set `SORA_VISION_MODEL` and,
when needed, `SORA_VISION_BASE_URL` / `SORA_VISION_API_KEY`. If no vision model
or OCR engine is available, the image is represented by its metadata. Each
image description makes one extra model request when a vision model is used.

Attachment and vision text is wrapped in an `<untrusted>` element with a random
per-turn boundary token. The system prompt tells the model that this content is
data, never instructions, and that identity/state edits may never be triggered
by it. DOCX/PPTX/XLSX and archive parsing never executes content and never
extracts nested archives.

### Adding a channel

Discovery imports only a tiny manifest per package, so a channel whose platform
SDK is missing does not break the rest (`discover_plugins` skips with a
warning):

```python
# sora_agent/channels/mychannel/manifest.py
from ..manifest import ChannelPlugin

PLUGIN = ChannelPlugin(
    name="mychannel",
    display_name="My Channel",
    runtime=f"{__package__}.runtime:MyChannel",
)

# sora_agent/channels/mychannel/runtime.py
from ..base import BaseChannel

class MyChannel(BaseChannel):
    name = "mychannel"
    async def start(self) -> None: ...   # poll/receive loop
    async def stop(self) -> None: ...    # idempotent
    async def send(self, msg) -> None: ...  # deliver one outbound message
```

`ChannelPlugin` validates names/runtime targets; `sora-gateway --list` prints
every discovered channel with its declared dependencies.

## Command-line interfaces

### `sora`

```
sora [--version]
sora setup [--mode recommended|advanced]
sora onboard [--mode recommended|advanced]     # alias for setup
sora chat
sora run TASK
sora gateway [start|run|stop|status|restart|logs]
sora gateway setup
sora doctor
sora update [--check] [--version REF] [--extras EXTRAS] [--repo OWNER/REPO]
sora set-key [provider|telegram]
sora config list
sora config get KEY
sora config set KEY VALUE
sora config unset KEY
```

Running `sora` with no command starts the local chat. If no configuration
exists, it offers the setup wizard first.

| Command | Purpose |
| --- | --- |
| `setup` | Run Recommended or Advanced Setup and write `config.json` + `secrets.env`. |
| `chat` | Local REPL with `/reset`, `/status`, `/eval`, `/help`, and `/exit`. |
| `run` | Execute one task with the configured provider, tools, and policies, then exit. |
| `gateway` | Run in the foreground, or use `start`, `stop`, `status`, `restart`, and `logs` to manage a detached process. `gateway setup` runs the wizard first. |
| `doctor` | Read-only check of paths, credentials, features, and safety warnings. |
| `update` | Upgrade an installer-managed environment or git checkout in place. `--check` only reports availability. |
| `set-key` | Store an API key or Telegram token in `secrets.env` without re-running setup. |
| `config list/get` | Inspect the saved configuration. `config get` uses dotted keys such as `provider.model`. |
| `config set/unset` | Change one value or remove it so the preset default applies again. Values are validated before saving. |

The config file is deliberately plain JSON:

```json
{
  "version": 1,
  "setup": {"mode": "recommended", "completed_at": "2026-11-05T00:00:00+00:00"},
  "agent": {
    "name": "Sora",
    "icon": "☁️",
    "personality": {"base": "friendly", "warmth": "very_warm"}
  },
  "provider": {
    "kind": "openai",
    "model": "gpt-5.4-mini",
    "temperature": 0.7,
    "context_window": 272000,
    "base_url": null
  },
  "channel": {"kind": "telegram", "allow_from": ["123456789"], "parse_mode": null},
  "features": {
    "memory": {"enabled": true, "sidecar": true, "auto_download": true},
    "files": {
      "enabled": true,
      "max_file_bytes": 20971520,
      "max_chars": 20000,
      "total_max_chars": 40000,
      "vision_model": null
    },
    "identity": true,
    "clock": true,
    "calendar": true,
    "sessions": true,
    "tools": "demo",
    "approval": "ask",
    "quality": true
  },
  "limits": {"max_steps": 12, "max_turns": 80},
  "paths": {"sessions_root": null, "calendar_db": null}
}
```

The example is abbreviated; `sora config list` prints the complete document.
`agent.icon` is used in terminal banners and signatures only. It is not
appended to every reply. `agent.personality` is prompt guidance, not an
automatic rewrite of `SOUL.md`.

Keys that come from the environment (for example `OPENAI_API_KEY` or
`SORA_TELEGRAM_TOKEN`) are stored separately in `secrets.env`. Environment
variables always override the file at runtime, even when setup saves a value.

### `sora-gateway`

```
sora-gateway [--channel CHANNEL] [--provider KIND] [--model ID]
             [--temperature F] [--max-steps N] [--max-turns N]
             [--no-sessions] [--provider-timeout S] [--provider-retries N]
             [--tool-timeout S] [--tools none|demo]
             [--memory] [--identity] [--approval off|ask]
             [--approval-timeout S] [--no-quality]
             [--no-clock] [--no-calendar] [--calendar-db PATH]
             [--token TOKEN] [--allow IDS] [--list]
```

A saved `sora setup` configuration supplies the defaults. With no config file,
the gateway keeps its original environment-only defaults, so existing
deployments behave exactly as before. Flags and `SORA_*` variables still win.

| Flag | No config file | Meaning |
| --- | --- | --- |
| `--channel` | `telegram` | Channel to run. With a config, `channel.kind` is the default. |
| `--provider` | `mock` | `mock`, `openai`, `deepseek`, `qwen`, or `custom`. |
| `--model` | provider default | Model id; overrides `OPENAI_MODEL` style variables. |
| `--temperature` | provider default | Sampling temperature; overrides `SORA_TEMPERATURE`. |
| `--max-steps` | `8` | Tool steps per turn. |
| `--max-turns` | `40` | Conversation turns kept per session. |
| `--no-sessions` | off | Keep history in memory only. |
| `--provider-timeout` | `30` | Provider request timeout (s). |
| `--provider-retries` | `2` | Provider retries. |
| `--tool-timeout` | `30` | Isolated tool process timeout (s). |
| `--tools` | `none` | `demo` registers the sample order tools. |
| `--memory` | off | Enable the CortexDB memory hook. |
| `--identity` | off | Inject the persistent USER.md/SOUL.md identity. |
| `--approval` | `ask` | `off` auto-approves consequential tools but still audits them. |
| `--approval-timeout` | `300` | Seconds to wait for a chat approval. |
| `--no-quality` | off | Disable post-run quality evaluation. |
| `--no-clock` | off | Omit the current-time section. |
| `--no-calendar` | off | Disable calendar tools. |
| `--calendar-db` | `~/.sora_agent/calendar.db` | Calendar database path. |
| `--token` / `--allow` | env | Override Telegram token / allow list. |
| `--list` | — | Print discovered channels and exit. |

`sora gateway` and `sora gateway run` are shortcuts for the same foreground
process when you want the saved configuration without retyping flags. Use
`sora gateway start` for the detached background process.

### `sora-memory`

```
sora-memory doctor [--json]              # configuration and health
sora-memory dream [--dry-run] [--today]  # transfer + consolidate
sora-memory backup [--active|--all|--daily YYYY-MM-DD ...]
sora-memory restore SNAPSHOT [--yes]     # restore from a snapshot
sora-memory recall QUERY [--scope auto|daily|longterm] [--k 8]
sora-memory daily-search QUERY [--k 8]   # verbatim search in daily files
sora-memory download | start | stop      # sidecar lifecycle
sora-memory optimize [--vacuum]          # offline SQLite maintenance
```

`sora-memory` refuses to run when memory is disabled (`SORA_MEMORY=0`).
`dream` prints the report path; `--dry-run` writes a full report without
mutating the store, and `--today` includes today's daily file.

## Environment variables

### Configuration and policy

Every saved setting is a default. These variables override it:

| Variable | Effect |
| --- | --- |
| `SORA_CONFIG` | Path to `config.json` (default `~/.sora_agent/config.json`). |
| `SORA_SECRETS` | Path to `secrets.env` (default `~/.sora_agent/secrets.env`). |
| `SORA_PROVIDER` | Provider kind: `mock`, `openai`, `deepseek`, `qwen`, or `custom`. |
| `SORA_TEMPERATURE` | Sampling temperature (`0.0`–`2.0`). |
| `SORA_CONTEXT_WINDOW` | Context-window budget in tokens (minimum `1000`). |
| `SORA_APPROVAL` | `ask` or `off`. Off still records audit entries. |
| `SORA_QUALITY` | `1`/`0` for post-run quality evaluation. |
| `SORA_CLOCK` | `1`/`0` for the current-time prompt section. |
| `SORA_CALENDAR` | `1`/`0` for native calendar tools. |
| `SORA_SESSIONS` | `1`/`0` for on-disk conversation history. |
| `SORA_TOOLS` | `none` or `demo`. |
| `SORA_PROVIDER_TIMEOUT` | Provider request timeout in seconds. |
| `SORA_FILES` | `1`/`0` for attachment-to-markdown understanding. |
| `SORA_FILES_MAX_BYTES` | Maximum attachment size in bytes. |
| `SORA_FILES_MAX_CHARS` | Maximum markdown characters per attachment. |
| `SORA_FILES_TOTAL_MAX_CHARS` | Maximum markdown characters for all attachments in one turn. |

Command-line flags override all of these. The same precedence applies to
`sora` and `sora-gateway`.

### Provider

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | `openai` / `azure` provider. Not used by other provider kinds. |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | `deepseek` provider. |
| `DASHSCOPE_API_KEY`, `QWEN_MODEL` | `qwen` provider. |
| `SORA_VISION_MODEL`, `SORA_VISION_BASE_URL`, `SORA_VISION_API_KEY` | Optional image-description model. Falls back to the main provider when it is known to accept images. |

### Subsystems

| Variable | Subsystem | Default |
| --- | --- | --- |
| `SORA_IDENTITY`, `SORA_IDENTITY_ROOT`, `SORA_IDENTITY_RETENTION`, `SORA_IDENTITY_MAX_REPAIRS` | Identity | on, `~/.sora_agent/identity`, 3, 2 |
| `SORA_MEMORY`, `SORA_MEMORY_ROOT`, and all `SORA_MEMORY_*` knobs | Memory | on, `~/.sora_agent` |
| `SORA_CALENDAR_DB` | Calendar | `~/.sora_agent/calendar.db` |
| `SORA_SESSIONS_ROOT` | Chat sessions | `~/.sora_agent/sessions` |
| `SORA_TELEGRAM_*` | Telegram channel | token required; deny-all allow list |
| `SORA_TELEGRAM_DOWNLOAD_FILES`, `SORA_TELEGRAM_MAX_ATTACHMENT_BYTES` | Telegram attachments | on, 20 MB |

The defaults in this table describe the library and a legacy environment-only
gateway. Once a `sora setup` config exists, its feature switches supply the
defaults and these variables override them.

See the [Memory](#memory-cortexdb) and [Identity](#identity-usermd--soulmd)
sections for the full tables.

## Project layout

```
SoraAgent/
├── install.sh                  # one-line installer for macOS/Linux
├── install.ps1                 # one-line installer for Windows
├── pyproject.toml              # packaging, extras, console scripts
├── pyrightconfig.json          # pyright in standard mode, Python 3.9
├── LICENSE                     # AGPL-3.0-only
├── README.md
├── examples/
│   ├── po_workflow.py          # interactive PO exception demo
│   ├── copilot_function.py     # Azure Function-style entry point
│   └── _demo_memory.py         # offline memory stand-in used by the examples
├── tests/                      # pytest suite (core, CLI, gateway)
└── sora_agent/
    ├── agent.py                # Agent facade + AgentResult
    ├── settings.py             # config schema, presets, secrets, precedence
    ├── runtime_setup.py        # build runtime objects from saved settings
    ├── model_catalog.py        # curated providers, models, context windows
    ├── persona.py              # agent name/icon/personality prompt layer
    ├── context.py              # deterministic context-window trimming
    ├── runtime.py              # per-run assembly (AgentRuntime, RunRequest)
    ├── runner.py               # pure model/tool loop (Runner, RunSpec, Step, events, metrics)
    ├── providers.py            # Message/ModelTurn/ToolCall, MockProvider, OpenAIChatProvider
    ├── files.py                # attachment-to-markdown conversion (FileUnderstander)
    ├── net.py                  # shared HTTPS/CA-bundle helper
    ├── tools.py                # Tool, ToolRegistry, ToolExecutor, ToolResult, @tool
    ├── tool_search.py          # ToolGate, FindToolsTool, two-tier lexical ranking
    ├── isolation.py            # spawned-process tool isolation with real timeouts
    ├── approval.py             # ApprovalGate + AuditEntry
    ├── eval.py                 # Trajectory, EvalReport, QualityGate, structural grader
    ├── judge.py                # optional calibrated trajectory judge
    ├── clock.py                # per-turn CURRENT TIME injection
    ├── calendar.py             # SQLite calendar + calendar tools
    ├── state.py                # session state document: store, monitor, protocol
    ├── state_provider.py       # StatefulProvider adapter
    ├── state_machine.py        # PO exception-handling state machine
    ├── identity.py             # USER.md/SOUL.md storage, validation, memory firewall
    ├── identity_provider.py    # identity prompt injection + edit handling
    ├── untrusted.py            # random-boundary framing for untrusted content
    ├── audit.py                # append-only audit log for automatic edits
    ├── rag.py                  # minimal Retriever protocol + in-memory stub
    ├── cli/                    # the sora command
    │   ├── main.py             # Typer app: setup, chat, gateway, config, doctor
    │   ├── setup.py            # Recommended and Advanced wizards
    │   ├── chat.py             # local REPL and one-shot run
    │   ├── doctor.py           # read-only health and safety check
    │   ├── ui.py               # Rich palette, panels, and tables
    │   └── prompts.py          # Questionary prompts with terminal fallback
    ├── channels/               # transports, bus, manager, chat handler, sessions, approvals
    │   ├── cli.py              # sora-gateway
    │   └── telegram/           # manifest + urllib transport + polling runtime
    └── memory/                 # CortexDB client/sidecar, daily store, recall, dream, backups
        └── cli.py              # sora-memory
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,memory,cli,files,dev]"

pytest                          # 538 tests
pyright                         # static checks, standard mode (Python 3.9; install separately)
```

Notes for contributors:

- `tests/conftest.py` disables default memory and identity
  (`SORA_MEMORY=0`, `SORA_IDENTITY=0`, `SORA_MEMORY_AUTO_DOWNLOAD=0`), points
  the calendar at a temp directory, and redirects `SORA_CONFIG` /
  `SORA_SECRETS`, so tests never touch the network or the real home directory.
- The codebase follows a strict philosophy: safety paths fail **closed**
  (approval, argument validation) while optional conveniences fail **open**
  (memory, identity, clock, calendar, observers, RAG).
- Evaluation and the judge are deliberately **non-executional** — do not wire a
  verdict into a tool call.
- Prefer standard library solutions; the core package has zero dependencies and
  the memory core, state core, and calendar are stdlib-only by design.

## License

GNU Affero General Public License v3.0 only (`AGPL-3.0-only`).

Copyright (C) 2026 Tam Kuan Hou (譚君浩 / Marco Tam) — GitHub:
[Mochi-Sora](https://github.com/Mochi-Sora).
