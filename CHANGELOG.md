# Changelog

All notable changes to SoraAgent are documented here.

## [0.1.3] - 2026-11-13

### Fixed

- Load configs written before 0.1.2, where `features.tools` held a tool-set
  name (`demo` or `none`) instead of a boolean. Those values now map to
  `true`/`false`, so upgrading no longer fails gateway startup with
  `expected a boolean, got 'demo'`. Existing installs can also edit
  `~/.sora_agent/config.json` directly; after this release no manual change
  is required.

## [0.1.2] - 2026-11-12

### Added

- **Built-in core tool pack** (`sora_agent/toolpack/`) with 12 hidden tools:
  `read_file`, `write_file`, `patch`, `search_files`, `terminal`,
  `process_manage`, `web_search`, `web_extract`, `todo_list`, `clarify`,
  `execute_code`, and `delegate_task`.
- `delegate_task` support in `Agent` and `ChatHandler`. A child agent receives
  the same provider and approval gate but no memory, identity, clock, calendar,
  or delegation tool, so it cannot recurse or read the parent conversation.
- Configurable clarification through a `ClarificationGate`; the terminal gets
  a console gate and headless surfaces fail soft with an `undelivered` result.
- Persistent `features.tools` setting, `SORA_TOOLPACK` switch, setup-wizard
  toggle, and `sora doctor` status row.
- `Tool(dedupe=True)` and per-run tool-call deduplication for idempotent
  consequential actions.
- `CHANGELOG.md`.

### Changed

- **Prompt ordering for cacheability:** the per-turn clock and session state
  are now trailing system messages after the conversation. The order is static
  system prompt → identity → conversation → state → clock.
- **Stable tool catalog:** the `find_tools` catalog no longer shrinks when a
  tool is loaded, so the tool-definition prefix remains stable across turns.
- **Action completion:** the system prompt now states that consequential tools
  are approval-gated and that the model must not substitute research for a
  requested action.
- **Memory retrieval:** daily search filters stopwords, tries AND before OR,
  and applies a recency tie-break. Long-term recall filters stopwords. The
  memory policy tells the model to retry with translated/original-language
  terms when a query is empty.
- Tool arguments are matched case-insensitively with ISO-8601 timestamp
  normalization in the benchmark scorer.
- Setup and doctor output expose the core tool pack.
- Recommended setup enables the core pack; Advanced setup starts with it off.

### Fixed

- Tool outputs and background-process pipes are closed cleanly, removing the
  unclosed-file `ResourceWarning`.
- Daily memory no longer treats common stopwords as OR terms, eliminating the
  no-answer false-positive behavior.
- Duplicate consequential calls no longer execute twice when a tool is marked
  idempotent.
- Calendar/action matching no longer fails solely because of title casing or
  equivalent ISO timestamp formatting.

### Validation

- 779 SoraAgent tests pass.
- Ruff lint and formatting checks pass.
- Full live benchmark on DeepSeek V4.1 Flash (`deepseek-flash`):
  overall 82.4/100, tool use 91.1, memory 94.8, dreaming 100, speed 74.0,
  autonomy 62.9, prompt caching 84.3.

## [0.1.1] - 2026-11-11

- Installer launches the setup wizard; release 0.1.1.
- Adopted running CortexDB sidecars across restarts.
- Restored token streaming and added the ten chat channel transports.

## [0.1.0] - 2026-10-26

- Initial public release.
