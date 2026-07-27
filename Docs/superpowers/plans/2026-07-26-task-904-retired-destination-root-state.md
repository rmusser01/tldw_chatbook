# Retired Destination Root State Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the remaining dead Notes, Search, retired Ingest-route, Tools,
and Evals root reactives and initializer paths while preserving their current
Library, Search, MCP, and Evals owners.

**Architecture:** Library retains Notes state and the Import media canvas reached
through the retired `ingest` alias; `SearchScreen`/its composed
`SearchRAGWindow`, MCP, and `EvalsScreen` retain their current destination
state. Root descriptors, dead handler-map strings, no-op watchers, timers, and
unused tab initializers are pure deletion; no replacement store or
compatibility property is introduced.

**Tech Stack:** Python 3.11+, Textual production screens, pytest/pytest-asyncio, AST ownership checks, Ruff.

**Backlog:** [TASK-904](../../../backlog/tasks/task-904%20-%20Remove-retired-Notes-Search-Ingest-Tools-and-Evals-root-state.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/033-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** Existing ADRs already assign view state to the registered destinations; this task deletes obsolete root mirrors.

---

## Execution and Test Boundary

Mounted checks go in
`Tests/ProductionApp/test_retired_destination_root_state.py`. One normal
`TldwCli` navigates through the actual Library, Search, MCP, and Evals screens,
including the `notes`, `ingest`, and `tools_settings` aliases at their current
Library/MCP owners. No destination shell or simplified application is allowed.

## Exact Removal Set

Remove the remaining names below. `evals_sidebar_collapsed` is already absent
on the latest `dev`; keep it in the ownership guard so it cannot return.

```text
current_selected_note_id
current_selected_note_version
current_selected_note_title
current_selected_note_content
notes_sort_by
notes_sort_ascending
notes_preview_mode
notes_auto_save_enabled
notes_auto_save_timer
notes_last_save_time
search_active_sub_tab
ingest_active_view
tools_settings_active_view
evals_sidebar_collapsed
_notes_search_timer
_initial_search_sub_tab_view
_initial_ingest_view
_initial_tools_settings_view
_activate_initial_ingest_view
```

## File Structure

- Modify `tldw_chatbook/app.py`: remove the exact fields, watchers, handlers, cleanup, constants, and imports.
- Keep the already-retired `tldw_chatbook/Event_Handlers/tab_initializers`
  package absent; do not recreate it to satisfy historical plan paths.
- Modify/delete `tldw_chatbook/Event_Handlers/note_ingest_events.py` or Notes compatibility paths only after a live-import census.
- Preserve destination state in `library_screen.py`, `search_screen.py`,
  `mcp_screen.py`, and `evals_screen.py`; change them only if a removed root
  access is found. The standalone Ingest screen was retired by TASK-684.4 and
  must not be recreated.
- Create `Tests/ProductionApp/test_retired_destination_root_state.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-904 and Add Exact Guards

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 904 -s "In Progress"
backlog task edit 904 --plan $'ADR required: yes\nADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs and the current route registry assign Notes and the retired Ingest route to Library, Search to SearchScreen, legacy Tools to MCP, and Evals to EvalsScreen.\n\n1. Add exact removed-name guards.\n2. Navigate every production destination and current legacy alias.\n3. Delete remaining root fields, timers, watchers, and initializers while keeping already-retired paths absent.\n4. Reconcile remaining stream task/plan IDs and latest-dev production routes, then verify destination-owned state remains.'
```

- [ ] Extend the AST guard for every exact name and dynamic string occurrence.
- [ ] Add mounted tests that navigate to `library`, `notes`, `ingest`,
  `search`, `mcp`, `tools_settings`, and `evals`; assert the exact current
  registered owner (`LibraryScreen`, `SearchScreen`, `MCPScreen`, or
  `EvalsScreen`); exercise one real destination-owned state change; navigate
  away/back; and verify no removed root access or compatibility property is
  used.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py -q
```

Expected: structural tests FAIL until deletion.

## Task 2: Delete Notes Root State and Timer Paths

- [ ] Remove the Notes descriptors, `_notes_search_timer`,
  `handle_notes_auto_save_toggle()`, its app event registration, and shutdown
  cleanup for the never-scheduled root timer.
- [ ] Keep Library's `_library_notes_sort`, editor/preview state, and autosave
  policy unchanged. Do not copy note bodies or selection into app/snapshot
  state.
- [ ] Remove legacy Notes ingestion exports only if no registered destination
  imports them; otherwise prune only root-state branches.

## Task 3: Delete Search/Ingest/Tools/Evals Root Paths

- [ ] Remove each descriptor, initial default, watcher, `watch_current_tab()`
  legacy branch, and dead initializer class/export.
- [ ] Remove `_activate_initial_ingest_view()` and its `call_later` sites.
- [ ] Keep:
  - `SearchScreen` actual `TabbedContent.active`;
  - Library's actual Import media canvas reached through the `ingest` alias;
  - MCP workbench mode;
  - Evals workbench state.
- [ ] Confirm the already-deleted initializer package still has no production
  import and remains covered by the legacy-entrypoint retirement guard.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

## Task 4: Verify and Close TASK-904

- [ ] Run:

```bash
python -m compileall -q tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py
python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py
python -m ruff format --check Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py
git diff --exit-code origin/dev -- tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py
git diff --check
```

Omit deleted optional paths from the final exact changed-file command; do not
restore dead code to satisfy a planned pathname. Do not mass-format the
verified pre-task `app.py` baseline exception or unchanged destination screens.

- [ ] Commit:

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py
git commit -m "refactor(state): remove retired destination root fields (task-904)"
```

- [ ] Re-read TASK-904, add Implementation Notes containing actual commands,
  counts, durations, modified/deleted files, ADRs, and deviations, check all
  acceptance criteria, then mark Done and commit its task file together with
  the reviewed stream-specification and plan reconciliation:

```bash
backlog task 904 --plain
backlog task edit 904 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 -s Done
git add Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md Docs/superpowers/plans/2026-07-26-task-653-retired-destination-root-state.md Docs/superpowers/plans/2026-07-26-task-654-tldw-api-result-envelope.md Docs/superpowers/plans/2026-07-26-task-655-reactive-ownership-closeout.md Docs/superpowers/plans/2026-07-26-task-904-retired-destination-root-state.md Docs/superpowers/plans/2026-07-26-task-905-retire-tldw-api-worker-pipeline.md Docs/superpowers/plans/2026-07-26-task-906-reactive-ownership-closeout.md 'backlog/tasks/task-904 - Remove-retired-Notes-Search-Ingest-Tools-and-Evals-root-state.md' 'backlog/tasks/task-905 - Replace-shared-TLDW-API-request-context-with-a-frozen-result-envelope.md' 'backlog/tasks/task-905 - Retire-unreachable-TLDW-API-worker-context-and-handlers.md'
git commit -m "docs(backlog): close retired destination state (task-904)"
```
