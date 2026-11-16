# Changelog

All notable changes to SoraAgent are documented here.

## [0.2.2] - 2026-11-16

### Added

- Recall now includes a bounded **Graph facts** section from CortexDB's native
  entity/relation graph. These are untrusted, redacted recall items and are
  capped by the normal recall budget.

### Documentation

- Clarified that SoraAgent's `dream:link` inferred two-hop records are transient
  diagnostics: they are generated and reported, then removed by artifact
  cleanup. Durable graph relationships come from CortexDB entities/relations.
- Clarified that `dream_community` labels are persisted as metadata but are not
  currently consumed by recall or clustering decisions.

## [0.2.1] - 2026-11-16

### Fixed

- CortexDB `get_memory` now treats HTTP 404 as a normal missing-memory lookup
  instead of raising. This was blocking every real daily transfer because the
  dream idempotency check could not query an absent id.
- CortexDB `delete_memory` now treats 404 as already deleted.
- Sidecar downloads verify the release archive checksum but pin the checksum of
  the extracted `cortexdb-grpc` binary. Previously the archive checksum was
  written as the executable pin, so the downloaded sidecar failed its own
  verification and could not start.
- Added regression coverage for the missing-memory lookup and binary pin.

## [0.2.0] - 2026-11-16

### Added

- Persistent specialists (delegation agents): a filesystem registry with
  manifests, lexical routing, `find_delegates` / `delegate_to_agent`, scoped
  tool grants, a dedicated workspace, read-only global memory recall,
  specialist daily stores, and specialist-to-worker spawning. Chat surfaces
  expose the two routing tools at the start of every turn.
- `sora delegate create/list/show/promote/pause/resume/delete`.
- Optional `features.delegation.create_mode: auto` (Advanced Setup, default
  off) for provisional, quota-limited specialist creation by the main agent.
- `Agent(system_prompt=...)` for specialist role prompts.
- Specialist daily stores now use the main SQLite `DailyStore`; dreams transfer
  them into unified long-term memory with provenance and half-done rounds.
- Retry/forget handling for failed daily transfers: two attempts on the first
  night, one retry on the next midnight, then a reported drop.
- Daily-only memory mode: when long-term memory is disabled or unavailable,
  daily stores are consolidated (dedupe + compact) and the previous snapshot is
  archived losslessly beside the store. No relationships/facts/decay run.
- Specialist daily backups under `memory backup/specialists/<id>/`; long-term
  memory now takes a post-consolidation snapshot as well.
- `sora setup` runs `sora doctor` after saving so unavailable long-term memory
  (and other selected features) is visible immediately.
- A document conversion pipeline in `sora_agent/documents/`: a converter
  registry, a shared `Document`/`Block` model, and Markdown rendering. The
  file understander is now a façade over it.
- ODT/ODS/ODP, EPUB, EML, and RTF adapters; ZIP/TAR conversion now recursively
  converts text/document members instead of only listing entries.
- Structured DOCX (headings, lists, tables), PPTX (slide tables, speaker
  notes), and XLSX (sheet names, formulas, merged-cell HTML tables) output.
- RTF text extraction and an actionable legacy-OLE capability message.
- `sora delegate` scoped workspace writes (`write_file` / `patch`) and
  durable daily notes via `remember_daily` / `recall_daily`.

### Changed

- `sora doctor` reports the active document converters, whether PDF text
  (`pypdf`) and image OCR are available, and the specialist registry.
- Desktop first-run defaults now register the core tool pack.
- Legacy `.doc/.xls/.ppt` and scanned PDFs are reported as capability gaps
  rather than silently rendered as binary/preview text.

### Fixed

- Specialist `remember_daily` passes through the identity redactor before the
  daily store write.
- Daily-store consolidation closes its SQLite connection; the full suite now
  passes with `-W error`.

## [0.1.5] - 2026-11-15

### Added

- Keyless DuckDuckGo search as the default `web_search` provider, with
  `duckduckgo`, `auto`, `tavily`, `brave`, `custom`, and `off` choices in
  `sora setup` and config.  Tavily and Brave keys are read from the
  environment or `secrets.env`, so daemon-launched gateways and bridges no
  longer need exported keys.
- Multiple workspace roots for the file tools (`SORA_TOOLPACK_ROOTS`) and an
  opt-in OS temp directory (`SORA_TOOLPACK_ALLOW_TMP`).

### Added

- Optional `http` (`requests`) and `scientific` (`numpy`) extras for
  `execute_code`. Recommended Setup never installs them; Advanced Setup asks
  before running pip, and `sora update --extras http,scientific` is the
  non-interactive path. `sora update` preserves them once present.

### Changed

- `build_registry()` accepts the secrets mapping; `sora doctor` now reports the
  workspace roots, web-search provider, Python version, optional code packages,
  and registered file tools.
- Workspace-boundary errors name the allowed roots and the variable that adds
  another directory.

## [0.1.4] - 2026-11-14

### Fixed

- Identity edit blocks (`<user_edit>` / `<soul_edit>`) are filtered from
  streamed deltas, so a live chat view no longer shows the raw block or
  presents the cleaned final answer as a duplicate second message.
- Identity prompts ask for unique bullet keys and a single line per
  multi-value fact, reducing rejected edits and repair turns.

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
