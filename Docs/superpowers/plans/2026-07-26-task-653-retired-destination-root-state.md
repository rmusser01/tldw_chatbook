# Retired Destination Root State Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the remaining dead Notes, Search, rebuilt Ingest, Tools, and Evals root reactives and initializer paths while preserving their registered destination owners.

**Architecture:** Library, `SearchRAGWindow`, `MediaIngestWindowRebuilt`, MCP, and `EvalsScreen` retain their current destination state. Root descriptors, dead handler-map strings, no-op watchers, timers, and unused tab initializers are pure deletion; no replacement store or compatibility property is introduced.

**Tech Stack:** Python 3.11+, Textual production screens, pytest/pytest-asyncio, AST ownership checks, Ruff.

**Backlog:** [TASK-653](../../../backlog/tasks/task-653%20-%20Remove-retired-Notes-Search-Ingest-Tools-and-Evals-root-state.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** Existing ADRs already assign view state to the registered destinations; this task deletes obsolete root mirrors.

---

## Execution and Test Boundary

Mounted checks go in
`Tests/ProductionApp/test_retired_destination_root_state.py`. One normal
`TldwCli` navigates through the actual Library, Search, Media Ingest, MCP, and
Evals screens. No destination shell or simplified application is allowed.

## Exact Removal Set

Remove:

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
- Modify/delete `tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py` and `__init__.py`: remove remaining dead Search/Ingest/Tools/Evals initializers; delete the package only if no live initializer remains.
- Modify/delete `tldw_chatbook/Event_Handlers/note_ingest_events.py` or Notes compatibility paths only after a live-import census.
- Preserve destination state in `library_screen.py`, `search_screen.py`, `media_ingest_screen.py`, `mcp_screen.py`, and `evals_screen.py`; change them only if a removed root access is found.
- Create `Tests/ProductionApp/test_retired_destination_root_state.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-653 and Add Exact Guards

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 653 -s "In Progress"
backlog task edit 653 --plan $'ADR required: yes\nADR path: backlog/decisions/026-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs assign these views to Library, Search, rebuilt Ingest, MCP, and Evals.\n\n1. Add exact removed-name guards.\n2. Navigate every production destination.\n3. Delete root fields, timers, watchers, and initializers.\n4. Verify destination-owned state remains.'
```

- [ ] Extend the AST guard for every exact name and dynamic string occurrence.
- [ ] Add mounted tests that navigate to:
  `library`, `search`, `ingest`, `mcp`, and `evals`; assert the exact
  registered screen class; exercise one real destination-owned state change;
  navigate away/back; and verify no removed root access or compatibility
  property is used.
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
  - `MediaIngestWindowRebuilt` actual tab;
  - MCP workbench mode;
  - Evals workbench state.
- [ ] If the initializer package becomes unreachable after TASK-647/651/652,
  delete it after `rg` proves no production import; otherwise leave only its
  live classes.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

## Task 4: Verify and Close TASK-653

- [ ] Run:

```bash
python -m compileall -q tldw_chatbook/app.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py
python -m ruff check tldw_chatbook/app.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py
git diff --check
```

Omit deleted optional paths from the final exact changed-file command; do not
restore dead code to satisfy a planned pathname. Do not mass-format the
verified pre-task `app.py` baseline exception.

- [ ] Commit:

```bash
git add tldw_chatbook/app.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py Tests/ProductionApp/test_retired_destination_root_state.py Tests/test_application_state_ownership.py
git commit -m "refactor(state): remove retired destination root fields (task-653)"
```

- [ ] Re-read TASK-653, add Implementation Notes containing actual commands,
  counts, durations, modified/deleted files, ADRs, and deviations, check all
  acceptance criteria, then mark Done and commit its task file:

```bash
backlog task 653 --plain
backlog task edit 653 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-653 - Remove-retired-Notes-Search-Ingest-Tools-and-Evals-root-state.md'
git commit -m "docs(backlog): close retired destination state (task-653)"
```
