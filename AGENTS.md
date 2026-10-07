# AGENTS.md

This file provides comprehensive guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**tldw_chatbook** - TUI application built with Textual for LLM interactions. Features: conversation management, character chat, notes with file sync, media ingestion, RAG capabilities.

**Tech Stack**: Python ≥3.12, Textual 8.x (≥8.0.0,<9), SQLite with FTS5, AGPLv3+
**Key Dependencies**: httpx, loguru, rich, pydantic, toml, keyring, aiofiles, jinja2

**Internals reference**: `Docs/Architecture/` holds one living, code-anchored doc per subsystem (Console, agent runtime, tools/MCP, RAG, DB layer, …) with an index in its README. Check there before assuming how a subsystem works.

## Quick Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"  # Or specific: .[embeddings_rag,websearch,local_vllm,ebook,pdf]

# Run
python3 -m tldw_chatbook.app

# Test
pytest  # All tests
pytest Tests/Chat/  # Specific module
pytest --cov=tldw_chatbook  # With coverage

# Note: The 'timeout' command is not available in this environment
```

## Architecture

### Core Structure

- **`app.py`** - Main entry, `TldwCli` class (composition root), navigation dispatch, global state
- **`config.py`** - TOML config at `~/.config/tldw_cli/config.toml`, env var fallbacks
- **`Constants.py`** - Route ids (TAB_CHAT, TAB_LIBRARY, …), display labels, navigation-context contract keys. UI dimensions are NOT here — they are `$ds-*` design tokens in `css/core/_variables.tcss` (ADR-150)

### UI Layer (`UI/` and `Widgets/`)

Navigation is screen-based and exclusive (no legacy tab-window UI): every destination is a `BaseAppScreen` (`UI/Navigation/base_app_screen.py`) registered lazily in `UI/Navigation/screen_registry.py`, with shell destinations + hotkeys defined in `UI/Navigation/shell_destinations.py`.

**Main Screens** (`UI/Screens/`):
- `chat_screen.py` - The Console: live agent work, streaming chat, images, RAG, approvals, tools
- `personas_screen.py` - Characters, personas, dictionaries, behavior profiles
- `library_screen.py` - Library hub: notes, media, conversations, prompts, skills, collections, search/RAG
- `evals_screen.py` - LLM benchmarking
- `settings_screen.py` - Settings (canonical surface)
- `watchlists_collections_screen.py`, `scheduling/schedules_workbench.py`, `llm_screen.py`, `image_gen_demo_screen.py`, and other feature screens

**Settings**: `UI/Screens/settings_screen.py` (F4 Settings destination) is the canonical settings surface. The legacy `UI/Tools_Settings_Window.py` is deprecated/unreachable — do not add new settings there (`Widgets/enhanced_settings_sidebar.py` no longer exists).

**Key Widgets**:
- `Widgets/Console/console_transcript.py` - Console transcript rendering (the live chat message widgets)
- `Widgets/Chat_Widgets/chat_message_enhanced.py` - Rich messages with actions (legacy Chat/CCP consumers; `Widgets/chat_message_enhanced.py` is a shim)
- `tool_message_widgets.py` - Tool-calling UI (ToolCallMessage, ToolResultMessage — Coding CoPilot only)
- `Widgets/NewIngest/` - Unified media ingestion UI (drop zone, processor, dashboard)

### Business Logic

- **`Chat/`** - `Chat_Functions.py` (conversation CRUD, `chat_api_call` provider dispatch), `console_chat_controller.py`/`console_chat_store.py` (Console engine), `document_generator.py` (LLM document generation)
- **`Character_Chat/`** - `Character_Chat_Lib.py` (card parse/persist, prompt composer), `character_card_formats.py` (multi-format detection). `ccv3_parser.py` is an empty placeholder — V3 cards ride the lenient V2 path in `Character_Chat_Lib.py`
- **`Notes/`** - `notes_sync_runtime.py` / `notes_sync_reconciler.py` / `notes_sync_executor.py` (review-first bidirectional sync — there is no single `sync_engine.py`), `file_notes_service.py` (session file-notes), template system
- **`RAG_Search/`** - `simplified/` for streamlined implementation, `chunking_service.py`, `ingestion_indexing.py` (shared service + daemon indexer)
- **`Tools/`** - `tool_executor.py` (always-on builtins: DateTimeTool, CalculatorTool), `local_tool_impls.py` (fs_* family)
- **`Evals/`** - `eval_orchestrator.py`, `eval_runner.py`, task-specific runners
- **`LLM_Calls/`** - Provider integrations: one `chat_with_<provider>()` per provider, dispatched via `Chat/Chat_Functions.py::chat_api_call` (`API_CALL_HANDLERS`). There is no unified `chat_with_provider()`
- **`Image_Generation/`** / **`Video_Generation/`** - Media-generation packages: adapter registry, config with secrets precedence, single validation choke point (`worker.run_generation`). Video mirrors image per ADR-044; `Video_Generation/video_store.py` is the ephemeral message-keyed video file store (task-3401.4); video backends land in task-3401.3/.6/.7

### Data Layer (`DB/`)

- **`ChaChaNotes_DB.py`** - Main DB (conversations, messages, characters, notes), schema v78 (`_CURRENT_SCHEMA_VERSION`; file-backed migrations in `DB/migrations/`)
- **`Client_Media_DB_v2.py`** - Media storage with chunking (in-code migration registry; Python-side FTS)
- **`RAG_Indexing_DB.py`** - Incremental-indexing state (item → last_modified); vectors live in ChromaDB, not here
- Other DBs: Evals, Prompts, Subscriptions, AgentRuns, Library collections/ingest-jobs, Workspaces
- Patterns: soft deletion, optimistic locking, FTS5 triggers, parameterized queries only

### Event System (`Event_Handlers/`)

Event flow: Widget → post_message() → @on() handler → workers → reactive updates → UI refresh

Live shared event modules: `TTS_Events/`, `STTS_Events/`, `media_events.py`, plus function-style handler modules (ingest, notes, LLM management, evals). **Note:** `Chat_Events/chat_messages.py`'s message catalog (ChatEvent, StreamingChunk, …) is defined but unwired — the live Console uses widget-local messages handled in `UI/Screens/chat_screen.py`; do not build on the dormant catalog.

### Key Patterns

**Database Operations**:
```python
with db.transaction() as cursor:
    cursor.execute(query, params)  # Always parameterized
```

**Reactive UI**:
```python
class MyWidget(Widget):
    data = reactive([], recompose=True)  # Rebuilds UI
    status = reactive("idle")  # Refresh only
```

**Background Work**:
```python
self.run_worker(self._heavy_task, exclusive=True)

@work(thread=True)
def _heavy_task(self):
    result = process()
    self.call_from_thread(self.update_ui, result)
```

## Development Guidelines

### Adding Features

**New LLM Provider**:
1. Add to `LLM_Calls/` as `chat_with_<provider>()` (streaming form returns an SSE generator)
2. Register in `Chat/Chat_Functions.py::API_CALL_HANDLERS` (+ `PROVIDER_PARAM_MAP` if params need projection)
3. Add config section; raise `ChatConfigurationError` on missing keys

**New Screen/Destination**:
1. Create a `BaseAppScreen` subclass in `UI/Screens/`
2. Add a route entry in `UI/Navigation/screen_registry.py` (`_SCREEN_ROUTES`; add a TAB_X constant if shared)
3. For a shell destination: extend `SHELL_DESTINATION_ORDER` + `SHELL_DESTINATION_SHORTCUTS` in `shell_destinations.py` (lockstep, ADR-031/152)
4. Add `@on(...)` handlers / workers on the screen

**New Tool**:
1. Extend the `Tool` class in `Tools/` (or add a `LocalToolSpec` in `Tools/local_tool_impls.py`)
2. Implement: get_name(), get_description(), get_parameters(), execute()
3. Register through the provider seam: a `ToolProvider` registered with `Agents/tool_catalog.py::ToolCatalogRegistry` (per-run composition happens in `Chat/console_agent_bridge.py`); gate with a `[tools]` config key — there is no `AVAILABLE_TOOLS` global

### Security Requirements

- Validate all inputs via `input_validation.py`
- Use `path_validation.py` for file paths
- SQL identifiers through `sql_validation.py`
- API keys from env/config only, never logged
- Sanitize HTML/Markdown content

### Performance Rules

- Workers for operations >100ms
- Stream LLM responses
- Chunk large files
- Paginate DB results
- Clear caches on context switch

### Configuration

Priority: env vars → config.toml → defaults

Key sections:
- `[api_settings]` (canonical provider settings; legacy `[openai_api]` etc. still overlaid)
- `[splash_screen]` - Animation settings
- `[Embeddings]` - Embedding provider/model (TOML section names are case-sensitive)
- `[AppRAGSearchConfig.rag.*]` - RAG engine config
- `[console]`, `[hooks]`, `[mcp]`, `[tools]`, `[agents]` - Console/agent behavior
- Provider-specific sections

### Testing

- **Ask before full sweeps**: before running the full test suite, ask whether a full sweep is wanted. Unless the user opts in, verify with targeted runs only — the tests touching the modified files/functionality
- Run the full suite only when the user asks for it (e.g. pre-PR/pre-merge verification they explicitly request)
- Use real SQLite in-memory for DB tests
- Property-based testing with Hypothesis
- Markers: unit, integration, optional, asyncio

### Merging into `dev`

(protection as of 2026-09-27): GitHub refuses a merge unless all of these hold:

- the required check is green;
- the branch is up to date with `dev` (strict);
- every review thread is resolved.

These apply to admins too.

The merge queue (`.github/workflows/merge-queue.yml`, spec
`Docs/superpowers/specs/2026-10-03-merge-queue-design.md`) runs in GitHub Actions. It works the same from any machine,
session or tool. Check its mode with `gh variable get MERGE_QUEUE`; if that prints an error (variable not found), the
variable is unset, which means off.

**Queue `on`:**
- Arm auto-merge (`gh pr merge <n> --auto --merge`) only when both of these hold:
  - Qodo has posted its review on the *current* head;
  - every thread on that head is resolved.

  Then walk away. The queue rebases the PR when it reaches the front, starts its CI, and lets auto-merge land it.
- Never `gh pr update-branch` an armed PR, and never merge an armed PR by hand. Either one makes the front PR's CI run go
  to waste.
- Before pushing more work to an armed PR, run `gh pr merge <n> --disable-auto`. Then `git pull --rebase`, because the
  queue may have rebased your branch, and push with `--force-with-lease`. Never use a plain force push.
- A conflict-free queue rebase needs no fresh Qodo review, because CI tests the combined result. New Qodo threads on the
  rebased head block the merge, and the queue evicts the PR with the reason.
- Never click "Approve and run" on a queue-rebased PR. Those runs are the token rebase's empty duplicates.
- An evicted PR has auto-merge off and a comment saying why. Fix the cause and re-arm; it rejoins at the back.

**Queue `off` or `dry`:**
- Re-sync a PR that is behind by **rebasing** it: `gh pr update-branch --rebase <n>`, or a local `git rebase origin/dev`
  pushed with `--force-with-lease`. Never merge `dev` into the branch (plain `gh pr update-branch` does exactly that).
- Re-sync only the PR you are about to merge. Under strict protection, one PR merges per CI cycle.
- Arm auto-merge only under the same Qodo and threads rule as above, and run `gh pr merge <n> --disable-auto` before
  pushing more work.

## Special Systems

### Tool Calling
- Schema v7 adds tool messages; `tool_executor.py` runs the builtin tools
- `Agents/tool_catalog.py` is the provider seam: builtin/local/skill/MCP providers register with one `ToolCatalogRegistry`
- Local fs_* tools (fs_list/fs_read/fs_write/fs_edit/fs_glob/fs_grep) in `Tools/local_tool_impls.py`, exposed via `Agents/local_tool_provider.py`
- Approvals flow through the MCP permission store; local tools sit under the `local:__local__` hub
- Console file authority: every live Chat gets private temporary scratch. Named Workspaces may add explicit folder bindings; local `fs_*`/Git uses scratch unless project instructions explicitly select one binding. `[console] workspace_root` is compatibility-only outside this Console path and never grants a Console Chat access. Workspace folder bindings may carry per-binding exclusions (exact user-marked paths, fully invisible to agent tools — see ADR-174).
- SSH workspace bindings reuse the Console file authority model: `ssh-filesystem` roots are `fs_*`-only (family B read_file/write_file/list_directory/edit_file stays local-only), status/availability per binding comes from the in-memory cache, and each call runs a transient stdlib worker over ssh (nothing installed on the server) — see ADR-181.

### Console Run Hooks
- User-configured external commands at six lifecycle events (UserPromptSubmit, PreToolUse, PostToolUse, ApprovalRequested, Stop, SubagentStop); config: `[hooks]` in config.toml (ADR-148).
- Deny-only: hooks can refuse tool calls but never bypass the permission store; PreToolUse fails closed, UserPromptSubmit fails open; argv-list commands only, process-group timeout kill.

### Console Project Instructions
- `AGENTS.override.md` / `AGENTS.md` startup and lazy nested guidance is untrusted, ephemeral user context bounded by one selected local-filesystem binding; it never grants tool permission.
- Registry ownership and path targets feed one shared activation ledger before normal tool review; read-only bindings do not advertise mutating tools.
- UI, persistence, and logs keep automatic bodies out of metadata surfaces; the explicit disposable Context **Next Send** preview is the only automatic UI body view.
- Governance: `backlog/decisions/069-console-project-instruction-local-state-and-preflight.md` and `Docs/superpowers/specs/2026-08-20-agents-md-support-design.md`.

### Config Encryption
- AES-256 with PBKDF2
- `Utils/config_encryption.py`
- Password dialogs in widgets

### Splash Screen
- 20+ animations in `Utils/Splash_Screens/` (by category; `Utils/splash_animations.py` is a compat stub)
- Config: `[splash_screen]` section
- Custom cards in `examples/custom_splash_cards/`

### Notes Sync
- Bidirectional file ↔ DB, review-first: both-sides-changed is always a surfaced conflict (KEEP_FILE/KEEP_NOTE/KEEP_BOTH), never auto-resolved — not last-write-wins
- Durable journaled execution; mid-apply crashes resume on next start
- Polling watcher (no FS-events dependency) drives reconciliation hints

### Pre-commit Hook
- `auto_review.py` for Codex integration
- Reviews diffs with LLM
- Exit 0 = pass, 2 = fail

### Model Catalog Auto-Refresh
- Startup background refresh of cloud-provider model lists (OpenAI, Anthropic, MistralAI, Moonshot, OpenRouter, ZAI) via `LLM_Provider_Catalog/`
- Disk TTL cache: `model_catalog_cache.json` in the user data dir (IDs + timestamps only)
- Capped merge (50) into model selectors; full catalog searchable in the Alt+M popover
- Config: `[model_catalog]` in config.toml; per-provider opt-in write-through appends new models to `[providers]`
- Governance: ADR-020 (amends ADR-002), spec/plan in Docs/superpowers/{specs,plans}/2026-07-17-model-catalog-auto-refresh*

### Workspace Assistant Defaults
- Explicit workspaces carry reference-backed `assistant_defaults` (persona + permission profile); Default/global stay unset.
- Persona policy rules narrow only (deny-by-default advertising, ask floors, per-run call caps); profiles inherit unset keys from `default`; all existing gates/floors apply first.
- Governance: `backlog/decisions/079-workspace-assistant-defaults.md` and `Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md`.

## Project-Specific Gotchas

1. **No localStorage** in artifacts - use React state or JS variables
2. **Tailwind limitations** - Only core utility classes, no compilation
3. **Schema migrations** - Always increment version, add to migrations/
4. **Optional deps** - Check with `optional_deps.py` before importing
5. **Thread safety** - Use transaction() context manager
6. **Route ids** - TAB_* constants must match `screen_registry.py` routes; shell destinations move in lockstep with their shortcuts (ADR-031/152)
7. **Streaming** - Always offer non-streaming fallback
8. **FTS5** - Triggers auto-update on text columns
9. **Workers** - Mark exclusive=True to prevent duplicates
10. **Reactive** - recompose=True rebuilds, default just refreshes
11. **Keybindings** - Screens must not bind terminal-convention keys (ctrl+c/v/x/s/d/z/a/r/w) or shadow the globals ctrl+p/ctrl+q/f1/f6; screen actions use single-letter htop-style bindings; footer hints may only advertise implemented actions — see `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`

## File Reference

Critical files for common tasks:
- Entry: `app.py`, `config.py`, `Constants.py`
- Chat/Console: `Chat/Chat_Functions.py`, `Chat/console_chat_controller.py`, `UI/Screens/chat_screen.py`
- DB: `base_db.py`, `ChaChaNotes_DB.py`
- LLM: `LLM_Calls/LLM_API_Calls.py`, `model_capabilities.py`
- Security: `path_validation.py`, `input_validation.py`
- UI: `Widgets/form_components.py`, reactive patterns in any widget
- Internals reference: `Docs/Architecture/README.md`

## Design Language (UI Tokens) — ADR-150

All UI work is governed by a design-token system. **Read
`backlog/docs/design-language.md` and `backlog/docs/component-patterns.md`
(ADR-161) before creating or modifying any screen, widget, or stylesheet.**

Hard rules:

- Every consistent visual value is a `$ds-*` token in
  `tldw_chatbook/css/core/_variables.tcss` — spacing (`$ds-space-*`),
  sizing (`$ds-control-*`), motion (`$ds-duration-*`), opacity, typography
  emphasis, colors, status, and component states. Do not invent literals or
  ad-hoc tokens in feature sheets; the governance test
  (`Tests/UI/test_design_token_governance.py`) fails on undefined `$ds-*`
  references and on new hex literals outside the token file.
- Describe new UI by composing tokens, component patterns, layout laws, and
  interaction rules from the constitution — never "make it look modern".
- Adding a value the language can't express? Add the token to
  `_variables.tcss` first, then use it.
- Never edit `tldw_chatbook/css/tldw_cli_modular.tcss` directly; edit source
  modules and rebuild with `python tldw_chatbook/css/build_css.py`.
- Python code assigns token-backed CSS classes instead of ad-hoc
  `styles.*` literals for token-covered values.
- Changing an existing token's value is a visual-breaking change; verify
  live per `backlog/docs/lessons-live-verification.md`.

## Code Style

- Type hints for public APIs
- Docstrings: Google style with Args/Returns/Raises
- Imports: stdlib → third-party → local
- PascalCase classes, snake_case functions
- Pydantic for validation
- Early returns to reduce nesting
- Constants for magic values
- Context managers for resources
- Descriptive test names
- Profile before optimizing
- Validate at boundaries
- Log errors with context

<!-- BACKLOG.MD GUIDELINES START -->
# Instructions for the usage of Backlog.md CLI Tool

## 1. Source of Truth

- Tasks live under **`backlog/tasks/`** (drafts under **`backlog/drafts/`**).
- Every implementation decision starts with reading the corresponding Markdown task file.
- Project documentation is in **`backlog/docs/`**.
- Project decisions are in **`backlog/decisions/`**.
- Hard-won working knowledge is in **`backlog/docs/lessons-*.md`** -- traps that have
  actually cost time in this repo, each recorded with the incident that produced it.
  **Read the one covering your area before starting**; they are short, and they exist
  because these mistakes recur:
  - `lessons-testing-evidence.md` -- what actually counts as evidence a change works
  - `lessons-live-verification.md` -- running the app and talking to a real server
  - `lessons-backlog-hygiene.md` -- task IDs, CLI quirks, git plumbing
- Canonical Architecture Decision Records (ADRs) live in **`backlog/decisions/`**. Historical ADR-like material elsewhere is reference-only unless a canonical ADR imports or supersedes it.
- Before implementation planning, read relevant ADRs and decide whether the task requires a new ADR.

## Architecture Decision Records (ADRs)

Use ADRs to record significant architectural decisions: what was decided, why, and what alternatives were considered.

An ADR is required for decisions involving storage/schema/migrations, sync/conflict policy, data ownership, provider or runtime boundaries, service contracts, cross-module interfaces, security/privacy/encryption/authentication, dependency/tooling/runtime choices, long-lived UX/application structure, or rejection of a plausible alternative future contributors may ask about again.

An ADR is not required for routine bug fixes, small UI polish, copy-only changes, mechanical refactors that preserve existing boundaries, test-only changes, or direct implementation of an existing ADR.

Every implementation plan must include:

```text
ADR required: yes/no
ADR path: backlog/decisions/NNN-short-title.md or N/A
Reason: brief explanation
```

If an ADR is required, create it before implementation begins and link it from the Backlog task, Superpowers plan, and final implementation notes. If an existing ADR applies, link it instead of creating a duplicate.

## 2. Defining Tasks

### Understand the Scope and the purpose

Ask questions to the user if something is not clear or ambiguous.
Break down the task into smaller, manageable parts if it is too large or complex.

### **Title (one liner)**

Use a clear brief title that summarizes the task.

### **Description**: (The **"why"**)

Provide a concise summary of the task purpose and its goal. Do not add implementation details here. It
should explain the purpose and context of the task. Code snippets should be avoided.

### **Acceptance Criteria**: (The **"what"**)

List specific, measurable outcomes that define what means to reach the goal from the description. Use checkboxes (
`- [ ]`) for tracking.
When defining `## Acceptance Criteria` for a task, focus on **outcomes, behaviors, and verifiable requirements** rather
than step-by-step implementation details.
Acceptance Criteria (AC) define *what* conditions must be met for the task to be considered complete.
They should be testable and confirm that the core purpose of the task is achieved.
**Key Principles for Good ACs:**

- **Outcome-Oriented:** Focus on the result, not the method.
- **Testable/Verifiable:** Each criterion should be something that can be objectively tested or verified.
- **Clear and Concise:** Unambiguous language.
- **Complete:** Collectively, ACs should cover the scope of the task.
- **User-Focused (where applicable):** Frame ACs from the perspective of the end-user or the system's external behavior.

    - *Good Example:* "- [ ] User can successfully log in with valid credentials."
    - *Good Example:* "- [ ] System processes 1000 requests per second without errors."
    - *Bad Example (Implementation Step):* "- [ ] Add a new function `handleLogin()` in `auth.ts`."

### Task file

Once a task is created it will be stored in `backlog/tasks/` directory as a Markdown file with the format
`task-<id> - <title>.md` (e.g. `task-42 - Add GraphQL resolver.md`).

### Task Breakdown Strategy

When breaking down features:

1. Identify the foundational components first
2. Create tasks in dependency order (foundations before features)
3. Ensure each task delivers value independently
4. Avoid creating tasks that block each other

### Additional task requirements

- Tasks must be **atomic** and **testable**. If a task is too large, break it down into smaller subtasks.
  Each task should represent a single unit of work that can be completed in a single PR.

- **Never** reference tasks that are to be done in the future or that are not yet created. You can only reference
  previous
  tasks (id < current task id).

- When creating multiple tasks, ensure they are **independent** and they do not depend on future tasks.   
  Example of wrong tasks splitting: task 1: "Add API endpoint for user data", task 2: "Define the user model and DB
  schema".  
  Example of correct tasks splitting: task 1: "Add system for handling API requests", task 2: "Add user model and DB
  schema", task 3: "Add API endpoint for user data".

## 4. Recommended Task Anatomy

```markdown
# task‑42 - Add GraphQL resolver

## Description (the why)

Short, imperative explanation of the goal of the task and why it is needed.

## Acceptance Criteria (the what)

- [ ] Resolver returns correct data for happy path
- [ ] Error response matches REST
- [ ] P95 latency ≤ 50 ms under 100 RPS

## Implementation Plan (the how) (added after putting the task in progress but before implementing any code change)

1. Research existing GraphQL resolver patterns
2. Implement basic resolver with error handling
3. Add performance monitoring
4. Write unit and integration tests
5. Benchmark performance under load

## Implementation Notes (imagine this is the PR description) (only added after finishing the code implementation of a task)

- Approach taken
- Features implemented or modified
- Technical decisions and trade-offs
- Modified or added files
```

## 5. Implementing Tasks

Mandatory sections for every task:

- **Implementation Plan**: (The **"how"**)  
  Outline the steps to achieve the task. Because the implementation details may
  change after the task is created, **the implementation plan must be added only after putting the task in progress**
  and before starting working on the task.
- **Implementation Notes**: (Imagine this is a PR note)  
  Start with a brief summary of what has been implemented. Document your approach, decisions, challenges, and any deviations from the plan. This
  section is added after you are done working on the task. It should summarize what you did and why you did it. Keep it
  concise but informative. Make it brief, explain ONLY the core changes and assume that others will read the code to understand the details.

**IMPORTANT**: Do not implement anything else that deviates from the **Acceptance Criteria**. If you need to
implement something that is not in the AC, update the AC first and then implement it or create a new task for it.

## 6. Typical Workflow

```bash
# 1 Identify work
backlog task list -s "To Do" --plain

# 2 Read details & documentation
backlog task 42 --plain
# Read relevant documentation files in `backlog/docs/`.
# Read relevant ADRs in `backlog/decisions/`; create/link one if the task makes a significant architectural decision.

# 3 Start work: assign yourself & move column
backlog task edit 42 -a @{yourself} -s "In Progress"

# 4 Add implementation plan before starting
backlog task edit 42 --plan "1. Analyze current implementation\n2. Identify bottlenecks\n3. Refactor in phases"

# 5 Break work down if needed by creating subtasks or additional tasks
backlog task create "Refactor DB layer" -p 42 -a @{yourself} -d "Description" --ac "Tests pass,Performance improved"

# 6 Complete and mark Done
backlog task edit 42 -s Done --notes "Implemented GraphQL resolver with error handling and performance monitoring"
```

### 7. Final Steps Before Marking a Task as Done

Always ensure you have:

1. ✅ Marked all acceptance criteria as completed (change `- [ ]` to `- [x]`)
2. ✅ Added an `## Implementation Notes` section documenting your approach
3. ✅ Run all tests and linting checks
4. ✅ Completed the ADR check: linked an existing ADR, created a new ADR, or documented why no ADR was required
5. ✅ Updated relevant documentation

## 8. Definition of Done (DoD)

A task is **Done** only when **ALL** of the following are complete:

1. **Acceptance criteria** checklist in the task file is fully checked (all `- [ ]` changed to `- [x]`).
2. **Implementation plan** was followed or deviations were documented in Implementation Notes.
3. **Automated tests** (unit + integration) cover new logic.
4. **Static analysis**: linter & formatter succeed.
5. **Documentation**:
    - All relevant docs updated (any relevant README file, backlog/docs, backlog/decisions, etc.).
    - Task file **MUST** have an `## Implementation Notes` section added summarising:
        - Approach taken
        - Features implemented or modified
        - Technical decisions and trade-offs
        - Modified or added files
6. **Review**: self review code.
7. **Task hygiene**: status set to **Done** via CLI (`backlog task edit <id> -s Done`).
8. **No regressions**: performance, security and licence checks green.
9. **Lessons learned**: if the task surfaced knowledge that generalises beyond it — a
   trap, a wrong assumption that cost time, a verification that only worked one way —
   add or update an entry in `backlog/docs/lessons-*.md`. **State the incident, not
   just the rule**: a lesson without the evidence that produced it decays into folklore
   and gets ignored. Most tasks produce nothing here, and that is fine; do not invent
   one to fill the slot.
10. **ADR hygiene**: ADR check completed; any new or superseded ADRs are linked from the task Implementation Plan and Implementation Notes.

⚠️ **IMPORTANT**: Never mark a task as Done without completing ALL items above.

## 9. Handy CLI Commands

| Action                  | Example                                                                                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Create task             | `backlog task create "Add OAuth System"`                                                                                                                      |
| Create with description | `backlog task create "Feature" -d "Add authentication system"`                                                                                                |
| Create with assignee    | `backlog task create "Feature" -a @sara`                                                                                                                      |
| Create with status      | `backlog task create "Feature" -s "In Progress"`                                                                                                              |
| Create with labels      | `backlog task create "Feature" -l auth,backend`                                                                                                               |
| Create with priority    | `backlog task create "Feature" --priority high`                                                                                                               |
| Create with plan        | `backlog task create "Feature" --plan "1. Research\n2. Implement"`                                                                                            |
| Create with AC          | `backlog task create "Feature" --ac "Must work,Must be tested"`                                                                                               |
| Create with notes       | `backlog task create "Feature" --notes "Started initial research"`                                                                                            |
| Create with deps        | `backlog task create "Feature" --dep task-1,task-2`                                                                                                           |
| Create sub task         | `backlog task create -p 14 "Add Login with Google"`                                                                                                           |
| Create (all options)    | `backlog task create "Feature" -d "Description" -a @sara -s "To Do" -l auth --priority high --ac "Must work" --notes "Initial setup done" --dep task-1 -p 14` |
| List tasks              | `backlog task list [-s <status>] [-a <assignee>] [-p <parent>]`                                                                                               |
| List by parent          | `backlog task list --parent 42` or `backlog task list -p task-42`                                                                                             |
| View detail             | `backlog task 7` (interactive UI, press 'E' to edit in editor)                                                                                                |
| View (AI mode)          | `backlog task 7 --plain`                                                                                                                                      |
| Edit                    | `backlog task edit 7 -a @sara -l auth,backend`                                                                                                                |
| Add plan                | `backlog task edit 7 --plan "Implementation approach"`                                                                                                        |
| Add AC                  | `backlog task edit 7 --ac "New criterion,Another one"`                                                                                                        |
| Add notes               | `backlog task edit 7 --notes "Completed X, working on Y"`                                                                                                     |
| Add deps                | `backlog task edit 7 --dep task-1 --dep task-2`                                                                                                               |
| Archive                 | `backlog task archive 7`                                                                                                                                      |
| Create draft            | `backlog task create "Feature" --draft`                                                                                                                       |
| Draft flow              | `backlog draft create "Spike GraphQL"` → `backlog draft promote 3.1`                                                                                          |
| Demote to draft         | `backlog task demote <id>`                                                                                                                                    |

Full help: `backlog --help`

## 10. Tips for AI Agents

- **Always use `--plain` flag** when listing or viewing tasks for AI-friendly text output instead of using Backlog.md
  interactive UI.
- When users mention to create a task, they mean to create a task using Backlog.md CLI tool.

<!-- BACKLOG.MD GUIDELINES END -->
