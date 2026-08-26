# Local Agent Tools (claude-code parity) — Design

**Date:** 2026-08-04
**Status:** Spec-reviewed (approved with advisories folded in); pending user review
**ADR required:** yes — new ADR in `backlog/decisions/` (extends the model-initiated permission/security boundary to local filesystem and network access; establishes a cross-module tool interface). Related: ADR-030 (`backlog/decisions/030-local-library-agent-tool-boundary.md`) for naming and result-size precedent.
**Related spec:** `Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md`

## 1. Purpose

Give the Console agent runtime a claude-code-style local tool set — file read/write/edit, glob, grep, web fetch/search, and session todos — registered through the existing `Agents/tool_catalog.py` provider seam, gated by the existing MCP approval machinery, and later exposed through the MCP server for external clients.

Explicit non-goals for this design: shell/bash execution (needs its own sandboxing spec), multi-edit, notebook editing, git tools, changes to the legacy `Tools/tool_executor.py` path beyond the shared-core refactor, and any new UI screens.

## 2. Tool inventory

Nine tools, named with the `fs_`/`web_`/`todo_` domain-prefix convention established by ADR-030 (`library_*`) and `mcp__server__tool`. The first three are migrations of existing legacy implementations.

| Tool | Origin | Behavior |
|---|---|---|
| `fs_read` | migrate `ReadFileTool` | Line-numbered output, `offset`/`limit` paging, 32 KiB byte-fitted cap (ADR-030), binary/image refusal with a clear message |
| `fs_write` | migrate `WriteFileTool` | Full-file create/overwrite, path-validated. Parent directory must exist — a deliberate divergence from claude-code's Write (which creates parents) to catch model path typos early |
| `fs_list` | migrate `ListDirectoryTool` | Depth-capped listing, entry-count cap |
| `fs_edit` | new | Exact `old_string`→`new_string` replacement; fails unless the match is unique (or `replace_all: true`). Ambiguity errors include the match count so the model can self-correct |
| `fs_glob` | new | Pure-Python pattern match under the workspace root, sorted by mtime, result-capped. No ripgrep dependency |
| `fs_grep` | new | Pure-Python regex scan with line numbers, context flags, count and files-with-matches modes, output caps |
| `web_fetch` | new | GET with redirect and size limits; HTML→text via `trafilatura.extract` (already a dependency, used in `Article_Extractor_Lib.py`) |
| `web_search` | migrate `WebSearchTool` | Delegates to `Web_Scraping/WebSearch_APIs.perform_websearch` (sync); wrapper supplies config defaults for its 15-arg dispatcher |
| `todo_write` | new | Session-scoped todo list (content/status/activeForm, claude-code `TodoWrite` semantics); session-lifetime, in-memory only |

> **Superseded for Console session tasks (TASK-13216, 2026-08-11).** This
> document preserves the historical `todo_write` design. The governing
> replacement is
> `Docs/superpowers/specs/2026-08-11-local-todo-task-api-design.md`: a supplied
> Console session store registers `todo_create`, `todo_update`, `todo_get`, and
> `todo_list`; stable session-local IDs and exact expected-version CAS protect
> atomic mutations, while pure task records and the next-ID high-water mark
> remain process-memory-only across in-process navigation. Unrelated file, web,
> and permission decisions in this historical design remain unchanged.

## 3. Architecture

### 3.1 Components

- **`Tools/local_tool_impls.py`** (new) — plain synchronous core functions, one per tool. The logic of the three legacy file tools moves here; the legacy async `Tool` wrappers and the new provider both call the same cores. `fs_edit`, `fs_glob`, `fs_grep`, `web_fetch`, `todo_write` are written here directly.
- **`Agents/local_tool_provider.py`** (new) — `LocalToolProvider` implementing the frozen `ToolProvider` protocol (`list_catalog`/`load_schema`/`invoke`), mirroring `BuiltinToolProvider`. Catalog ids are `local:<tool_name>`, source `"local"`. Sync `invoke`, worker-thread safe, never touches Textual or the event loop.
- **`Chat/console_agent_bridge.py`** — compose the local provider alongside Builtin + Skill + MCP providers per run.
- **Approval generalization** — a `build_local_review_hook` parallel to `build_mcp_review_hook` (`Chat/console_chat_controller.py:70`), reusing the same `request_*_approvals` card round-trip (`chat_approval_card.py`: Approve once / for session / Always allow / Deny) and the same permission store.

### 3.2 Permission model

Local tools join `MCP/permission_store.py` under a synthetic server key (`local:__local__`, distinct from the existing `builtin:tldw_chatbook` no-transport precedent). No store schema change — server keys are opaque. Precedence (tool override → server default → global default, default `ask`), session approvals, kill switch, and rug-pull hashing all apply unchanged.

- `fs_write`, `fs_edit`, `todo_write` carry the `mutates` high-risk tag. Tags are synthesized on the `HubTool` view used for permission resolution; `ToolCatalogEntry` itself has no tag field. The risk floor downgrades only *inherited* allows to ask; an explicit tool-level "Always allow" is never floored, so the approval card offers "Always allow" for these tools exactly as it does for MCP tools (MCP parity), and that choice sticks.
- `web_fetch`, `web_search` are network-classed (default `ask`).
- `fs_read`, `fs_glob`, `fs_grep`, `fs_list` follow the global default.
- `always_allow` requires a `definition_hash(description, input_schema)` (`permission_store.py:385`) even for static local tools; the provider computes it at startup.

### 3.3 Approval correctness discipline (three mechanisms, all required)

MCP's fail-closed behavior is not in the hook; a local provider must replicate all three:

1. **Clear-first**: the review hook calls `apply_batch_decisions({})` at entry so a hook exception (which fails open in `agent_runtime.py:366-376`) cannot reuse stale stamps.
2. **Fail-closed invoke**: `invoke()` refuses when no per-turn stamp exists, using these pinned single-sourced constants (tests assert on them verbatim):
   - `LOCAL_DENY_REFUSAL = "blocked by local tool permissions (set to Off)"`
   - `LOCAL_TIMEOUT_REFUSAL = "user did not approve within the time limit; do not retry"`
   - `LOCAL_KILL_SWITCH_REFUSAL = "blocked — local tools are switched off"`
3. **`review_state_scope`** snapshot/restore around nested sub-agent runs (`agent_service.py:133-165`) so child runs cannot clobber parent verdicts.

### 3.4 Path policy

The legacy tools' `validate_path(file_path, "file")` calls are **not** reusable: they confine to `<cwd>/file` (a root that almost never exists) and reject any hidden path component, which breaks real repos (`.github/`). New policy:

- A real workspace root: `[console] workspace_root` config (default: app cwd at startup), coerced and templated following the `collapse_large_pastes` precedent (`config.py:770-790`, template `config.py:2134-2136`).
- All path-taking tools resolve against the workspace root via `Utils/path_validation.validate_path`; anything outside is refused **before** approval is consulted.
- Hidden components are allowed **under** the workspace root (the dotfile ban does not apply to in-root paths).

### 3.5 Registration, disclosure, and budgets

- The registry is first-registrant-wins on name collisions (`tool_catalog.py:311-323`); the `fs_`/`web_`/`todo_` prefixes avoid collisions with legacy `Tools/` names, `library_*`, and `mcp__*`.
- Adding nine tools will push most runs past `DIRECT_DISCLOSE_THRESHOLD = 8` (`agent_models.py:35`), flipping disclosure from direct to `find_tools`/`load_tools` discovery. This is the designed progressive-disclosure behavior; the spec's tests must exercise the find/load path so no tool is silently unreachable. `RunBudget.max_active_tools` stays at the Console default (8).

### 3.6 Result discipline

- Tool results are byte-fitted to 32 KiB (ADR-030), errors short and exact. The MCP 4,000-char cap does not apply to this provider; note that sub-agent result relay still truncates at 4,000 (`agent_service.py:438-440`).
- The provider never raises across the boundary: every failure (file not found, ambiguous match with count, path outside workspace, fetch error, approval refusal) returns as a tool-result string the model can act on.

### 3.7 Todo state

One new field on the mutable `ConsoleChatSession` dataclass (`console_chat_store.py:123-134`) holds the session's todo list in memory. `todo_write` mutates it and posts a message so the transcript renders the current list. No durable persistence in this design (matches claude-code's session-scoped `TodoWrite`).

### 3.8 MCP exposure (phase 4)

The same nine tools are added to `MCP/server.py` backed by the same `local_tool_impls.py` cores, giving external MCP clients parity. This is a separate backlog task after the Console phases land.

## 4. Data flow

Model emits tool call → `agent_runtime.run_agent_loop` → per-batch `review_tool_calls` hook (clear stamps → approval card via `app.call_from_thread` + `threading.Event`, 120 s timeout → verdicts stamped) → `ToolCatalogRegistry.invoke` → `LocalToolProvider.invoke` (no-stamp → exact refusal string) → sync core in `Tools/local_tool_impls.py` → result byte-fitted to 32 KiB → back to the model; UI renders via existing `ToolCallMessage`/`ToolResultMessage` widgets.

## 5. Error handling

- Approval outcomes use single-sourced refusal-string constants; deny/timeout/kill-switch semantics identical to MCP.
- Tool-execution failures return as result strings (never exceptions across the provider boundary): `fs_edit` ambiguity reports match count; path refusals name the workspace root; `web_fetch` reports HTTP status/size-limit/redirect-limit distinctly.
- "Ask" with no approval callback configured fails closed (established MCP semantic).

## 6. Testing

- **Unit** (`Tests/Tools/`): each core against `tmp_path` — read paging/caps, edit unique-match/replace_all/ambiguity, glob mtime ordering and caps, grep modes, `web_fetch` with `httpx.MockTransport` (redirects, size limits, trafilatura extraction), todo state transitions.
- **Provider** (`Tests/Agents/`): catalog listing and ids (`local:fs_read`, …), schema shapes, byte-fitting, never-raises boundary across every exception class.
- **Approval gate**: allow / approve-once / approve-session / always (with definition hash) / deny / kill-switch / fail-closed-without-callback; stale-stamp bypass after hook exception; `review_state_scope` sub-agent isolation.
- **Integration**: agent runtime loop with the fence-first protocol against a fake model emitting tool calls, verifying dispatch, refusals, and the `find_tools`/`load_tools` discovery path once the catalog exceeds the direct-disclose threshold.
- **Property** (Hypothesis, already a project dependency): `fs_edit` matching (uniqueness, replacement invariants) and workspace-root confinement (no escape via `..`, symlinks, hidden components).

## 7. Phasing (each its own backlog task)

1. **Plumbing**: `local_tool_impls.py` scaffold, `LocalToolProvider`, `build_local_review_hook` + permission-store wiring + definition hashes, workspace-root config, bridge registration, `[console] local_tools_enabled` master flag.
2. **File tools**: `fs_read`/`fs_write`/`fs_list` migration + `fs_edit` + `fs_glob` + `fs_grep`.
3. **Research tools**: `web_fetch` + `web_search` migration + `todo_write` with transcript rendering.
4. **MCP exposure**: same nine tools on `MCP/server.py` via the shared cores.

## 8. Documentation updates

- New ADR in `backlog/decisions/` covering the local-tool permission boundary and naming convention; linked from all four tasks.
- `AGENTS.md`: update "Special Systems → Tool Calling" (execution is no longer pending; describe the catalog/provider seam); correct stale `Coding_Window.py`/`TAB_CODING` references.
- Config template + `[console]` section docs for `workspace_root` and `local_tools_enabled`.
