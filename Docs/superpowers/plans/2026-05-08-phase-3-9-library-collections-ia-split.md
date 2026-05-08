# Phase 3.9 Library Collections IA Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the combined `W+C` product model into a Watchlists-only top-level destination and a Library-owned local Collections management workflow.

**Architecture:** Preserve compatibility route IDs while changing user-facing labels and copy to `Watchlists`. Add Library-owned pure state, local SQLite persistence, service contracts, and a Textual Collections panel inside the existing Library shell. Do not reuse Watchlist or read-it-later services as the Collection model, and do not expose fake server sync before the sync engine exists.

**Tech Stack:** Python 3.12, Textual, pytest, Backlog.md, SQLite, existing `LibraryScreen`, `WatchlistsCollectionsScreen`, shell destination metadata, Home active-work adapter, Console live-work state, and Library widget patterns.

---

## Source Of Truth

- Design spec: `Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md`
- Current roadmap: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Parent backlog task: `TASK-10.9`
- Child execution slices: `TASK-10.9.1` through `TASK-10.9.4`
- Baseline branch: `origin/dev` at `df63b49f` plus design commit `9226ff47`

## Spec Review Result

Self-review found and fixed two issues before this plan was written:

- The new spec used non-ASCII box drawing. It now uses ASCII layout art.
- The spec did not explicitly cover Home and Console source labels during the `W+C` split. It now requires user-facing active-work labels to say `Watchlists`, while allowing internal compatibility payload values where needed.

No blocking design gap remains. The main implementation risk is breadth: `W+C` appears in navigation, command palette help, Watchlists destination copy, Home active-work messages, Console readiness copy, and existing tests. Task 1 must update the visible model without breaking the `watchlists_collections` route.

## Scope

Included:

- Visible `W+C` / `Watchlists+Collections` shell copy becomes `Watchlists`.
- `watchlists_collections` route ID remains compatible.
- Watchlists destination stops loading or rendering Collections/read-it-later data.
- Home and Console visible active-work copy says `Watchlists`.
- Library Collections mode supports local list, create, select, rename, and delete.
- Collection state shows `local-only` or `sync-unavailable` status, not fake sync controls.
- Citations/snippets remain tracked as later-stage Search/RAG features.
- QA evidence and roadmap/backlog tracking close the gate only after a usable walkthrough.

Excluded:

- Server sync engine.
- Full server parity for Collections.
- Full cross-source item picker.
- Collection-scoped RAG, Study, Flashcards, Quizzes, Console, or Import/Export execution.
- Deleting the `watchlists_collections` route or class names in this gate.
- Broad Watchlists runtime rewrite.

## File Structure

### Create

- `tldw_chatbook/DB/Library_Collections_DB.py`
  - Local SQLite persistence for Library collection records and future item membership counts.
- `tldw_chatbook/Library/library_collections_state.py`
  - Pure display-state models for summaries, detail, action readiness, and panel status.
- `tldw_chatbook/Library/library_collections_service.py`
  - Service protocol and local service adapter for list/get/create/rename/delete.
- `tldw_chatbook/Widgets/Library/library_collections_panel.py`
  - Textual widget for Library Collections list, detail, form controls, and inspector state.
- `Tests/Library/test_library_collections_state.py`
- `Tests/Library/test_library_collections_service.py`
- `Tests/UI/test_product_maturity_phase39_library_collections.py`
- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-phase-3-9-library-collections.md`

### Modify

- `tldw_chatbook/Constants.py`
  - Change display label for `TAB_WATCHLISTS_COLLECTIONS` from `W+C` to `Watchlists`.
- `tldw_chatbook/app.py`
  - Update `TabNavigationProvider.TAB_HELP_TEXT`.
  - Wire `library_collections_service` after DB/config helpers exist.
- `tldw_chatbook/config.py`
  - Add `library_collections_db_path` config default and `get_library_collections_db_path()`.
- `tldw_chatbook/UI/Navigation/shell_destinations.py`
  - Make the destination user-facing label/full label/purpose/tooltip Watchlists-only.
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
  - Keep compatibility class/route IDs, but render Watchlists-only copy and data.
- `tldw_chatbook/Home/active_work_adapter.py`
  - Update visible messages, source labels, recovery, and action labels from `W+C` to `Watchlists`.
- `tldw_chatbook/Chat/console_live_work.py`
  - Update live-work readiness labels and recovery copy.
- `tldw_chatbook/Chat/console_display_state.py`
  - Update staged-context recovery copy from `W+C` to `Watchlists`.
- `tldw_chatbook/UI/Screens/library_screen.py`
  - Mount and handle Library Collections panel when active mode is `collections`.
- `tldw_chatbook/Widgets/Library/__init__.py`
  - Export the new Collections widget.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add source styles only if new Collections widgets need shared hooks.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate only if source TCSS changes.
- `Tests/UI/test_shell_destinations.py`
- `Tests/UI/test_master_shell_navigation.py`
- `Tests/UI/test_command_palette_providers.py`
- `Tests/UI/test_destination_shells.py`
- `Tests/UI/test_console_live_work_handoffs.py`
- `Tests/UI/test_unified_shell_phase6_first_time_replay.py`
- `Tests/UI/test_unified_shell_phase6_power_user_replay.py`
- `Tests/Home/test_active_work_adapter.py`
- `Tests/Home/test_dashboard_state.py`
- `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- `backlog/tasks/task-10.9*.md`

## Read Before Editing

- `Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md`
- `Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md`
- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `tldw_chatbook/UI/Navigation/shell_destinations.py`
- `tldw_chatbook/Home/active_work_adapter.py`
- `tldw_chatbook/Chat/console_live_work.py`
- `tldw_chatbook/config.py`
- `Tests/UI/test_destination_shells.py`
- `Tests/UI/test_product_maturity_gate16_library_search_rag.py`
- `backlog/tasks/task-10.9 - Product-Maturity-Phase-3.9-Library-Collections-IA-Split.md`

Use the repo virtualenv from any worktree:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

## Risk Controls

- Keep `watchlists_collections` as the route ID for this gate.
- Avoid changing payload routing IDs unless tests prove the compatibility path still works.
- Do not reuse `media_reading_scope_service.list_read_it_later()` as the Library Collections data model.
- Do not add enabled RAG/Study/Console actions for Collections until item membership and source-scoped execution are implemented.
- Do not edit generated CSS directly. Edit source TCSS first, then run `python tldw_chatbook/css/build_css.py` if needed.
- Avoid fixed `pilot.pause()` sleeps in new UI tests. Poll for selectors, enabled state, or stable visible text.
- Prefer selector/action-state assertions over brittle full-paragraph copy checks.

---

### Task 1: Phase 3.9.1 Watchlists IA Split And Compatibility Labels

Backlog: `TASK-10.9.1`

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/Home/active_work_adapter.py`
- Modify: `tldw_chatbook/Chat/console_live_work.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Test: `Tests/UI/test_shell_destinations.py`
- Test: `Tests/UI/test_master_shell_navigation.py`
- Test: `Tests/UI/test_command_palette_providers.py`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/Home/test_active_work_adapter.py`
- Test: `Tests/UI/test_console_live_work_handoffs.py`

- [ ] **Step 1: Run the current Watchlists/W+C baseline**

```bash
$PY -m pytest -q \
  Tests/UI/test_shell_destinations.py \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_destination_shells.py::test_watchlists_collections_uses_compact_title_and_clear_sections \
  Tests/UI/test_destination_shells.py::test_watchlists_collections_lists_local_snapshot_from_services \
  Tests/Home/test_active_work_adapter.py \
  --tb=short
```

Expected: pass on baseline before red tests are changed.

- [ ] **Step 2: Add red tests for user-facing Watchlists copy**

Update shell/navigation tests to expect:

```python
wc = get_shell_destination("watchlists_collections")
assert wc.label == "Watchlists"
assert wc.full_label == "Watchlists"
assert "Collections" not in wc.tooltip
```

Update command palette tests to search for `Watchlists` and assert `Watchlists+Collections` is not in command/help text.

Update destination tests to assert:

```python
assert _static_text(screen.query_one("#watchlists-collections-title", Static)) == "Watchlists"
assert "Collections" not in _visible_text(screen)
assert "Local Watchlists snapshot" in _visible_text(screen)
```

Expected before implementation: failures in navigation metadata, command palette help, and Watchlists screen body.

- [ ] **Step 3: Update shell metadata and command palette labels**

Change:

- `Constants.TAB_DISPLAY_LABELS[TAB_WATCHLISTS_COLLECTIONS]` to `Watchlists`.
- `TabNavigationProvider.TAB_HELP_TEXT[TAB_WATCHLISTS_COLLECTIONS]` to Watchlists-only copy.
- `SHELL_DESTINATION_ORDER` entry for `watchlists_collections` to:

```python
ShellDestination(
    "watchlists_collections",
    "Watchlists",
    "watchlists_collections",
    "Monitored sources, runs, alerts, and recovery.",
    "Open Watchlists for monitored sources, runs, alerts, and recovery.",
    ("subscriptions", "subscription"),
    full_label="Watchlists",
    navigation_priority=40,
)
```

- [ ] **Step 4: Make `WatchlistsCollectionsScreen` render Watchlists-only content**

Keep the class name and `screen_id` for compatibility, but remove collection data from the visible snapshot and UI.

Implementation notes:

- Rename constants to Watchlists wording where practical.
- Stop calling `media_reading_scope_service.list_read_it_later()` inside `_list_local_wc_snapshot()`.
- Return only watchlist records/counts from the local snapshot.
- Keep selector IDs such as `#watchlists-collections-shell` if existing visual audit tests rely on them.
- Remove `#wc-collections-summary` and `#wc-collection-item-*` from composed output.
- Update `ChatHandoffPayload` title/body/display summary/suggested prompt to Watchlists-only copy.
- Keep `source="watchlists_collections"` if downstream tests require the compatibility source route.

- [ ] **Step 5: Update Home and Console visible active-work labels**

Change user-facing strings:

- `Opening W+C run details` -> `Opening Watchlists run details`
- `source="W+C"` -> prefer `source="Watchlists"` for new active-work items
- `Review the W+C run details or retry from W+C` -> Watchlists wording
- `Open W+C run` -> `Open Watchlists run`
- Console readiness row label `W+C` -> `Watchlists`

Compatibility guard:

- Where code filters by `item.source == "W+C"`, accept both `Watchlists` and `W+C` during this gate.

- [ ] **Step 6: Verify Task 1**

```bash
$PY -m pytest -q \
  Tests/UI/test_shell_destinations.py \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_destination_shells.py \
  Tests/Home/test_active_work_adapter.py \
  Tests/UI/test_console_live_work_handoffs.py \
  --tb=short

git diff --check
```

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  tldw_chatbook/UI/Navigation/shell_destinations.py \
  tldw_chatbook/Constants.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  tldw_chatbook/Home/active_work_adapter.py \
  tldw_chatbook/Chat/console_live_work.py \
  tldw_chatbook/Chat/console_display_state.py \
  Tests/UI/test_shell_destinations.py \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_destination_shells.py \
  Tests/Home/test_active_work_adapter.py \
  Tests/UI/test_console_live_work_handoffs.py
git commit -m "Split Watchlists IA from Collections"
```

---

### Task 2: Phase 3.9.2 Library Collections Display-State And Local Service Contracts

Backlog: `TASK-10.9.2`

**Files:**
- Create: `tldw_chatbook/DB/Library_Collections_DB.py`
- Create: `tldw_chatbook/Library/library_collections_state.py`
- Create: `tldw_chatbook/Library/library_collections_service.py`
- Modify: `tldw_chatbook/Library/__init__.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Library/test_library_collections_state.py`
- Create: `Tests/Library/test_library_collections_service.py`

- [ ] **Step 1: Add pure state red tests**

Create `Tests/Library/test_library_collections_state.py` covering:

- Empty panel state shows the copy `Group saved Library items for Search/RAG, Study, and Console.`
- Ready state selects the first collection by default when no selected ID is provided.
- Invalid create/rename input disables actions with a visible reason.
- Sync status renders `local-only` or `sync-unavailable`.
- Delete action is disabled when no collection is selected.

Expected before implementation: import failure for `tldw_chatbook.Library.library_collections_state`.

- [ ] **Step 2: Add service/DB red tests**

Create `Tests/Library/test_library_collections_service.py` using a temp SQLite path.

Cover:

- `list_collections()` returns an empty list initially.
- `create_collection("Research")` persists a collection with stable `collection_id`, `item_count == 0`, `source_authority == "local"`, and `sync_status == "local-only"`.
- Duplicate normalized names are rejected.
- `rename_collection()` updates `name`, optional `description`, and `updated_at`.
- `delete_collection()` hides/deletes the record from list/get.
- Invalid names are rejected before SQL.

Expected before implementation: import failure for `tldw_chatbook.Library.library_collections_service`.

- [ ] **Step 3: Implement local SQLite persistence**

Create `tldw_chatbook/DB/Library_Collections_DB.py` with:

```sql
CREATE TABLE IF NOT EXISTS library_collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS library_collection_items (
    item_id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(collection_id) REFERENCES library_collections(collection_id)
);
```

Keep item membership private/future-facing in this gate. It exists so `item_count` can be computed without changing the persistence contract later.

- [ ] **Step 4: Implement pure state models**

Create frozen dataclasses:

- `LibraryCollectionSummary`
- `LibraryCollectionDetail`
- `LibraryCollectionActionState`
- `LibraryCollectionsPanelState`

Recommended builder:

```python
LibraryCollectionsPanelState.from_values(
    collections=records,
    selected_collection_id=selected_id,
    status="ready",
    error_message="",
)
```

The state builder should sanitize display strings and never raise on loose app/test seam values.

- [ ] **Step 5: Implement service contract**

Create:

- `LibraryCollectionsService` protocol.
- `LocalLibraryCollectionsService` implementation.
- `LibraryCollectionsServiceError`.
- `InvalidLibraryCollectionName`.
- `DuplicateLibraryCollectionName`.

Validation:

- Trim names.
- Reject empty names.
- Limit names to 120 characters.
- Limit descriptions to 500 characters.
- Use parameterized SQL only.

- [ ] **Step 6: Add config/app wiring**

In `config.py`, add:

- default `library_collections_db_path = "~/.local/share/tldw_cli/tldw_chatbook_library_collections.db"`
- `get_library_collections_db_path()`

In `app.py`, wire:

```python
self.local_library_collections_service = LocalLibraryCollectionsService(
    LibraryCollectionsDB(get_library_collections_db_path())
)
self.library_collections_service = self.local_library_collections_service
```

If initialization fails, set `self.library_collections_service = None` and rely on the Library UI recovery state.

- [ ] **Step 7: Verify Task 2**

```bash
$PY -m pytest -q \
  Tests/Library/test_library_collections_state.py \
  Tests/Library/test_library_collections_service.py \
  --tb=short

git diff --check
```

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  tldw_chatbook/DB/Library_Collections_DB.py \
  tldw_chatbook/Library/__init__.py \
  tldw_chatbook/Library/library_collections_state.py \
  tldw_chatbook/Library/library_collections_service.py \
  tldw_chatbook/config.py \
  tldw_chatbook/app.py \
  Tests/Library/test_library_collections_state.py \
  Tests/Library/test_library_collections_service.py
git commit -m "Add Library Collections local service contracts"
```

---

### Task 3: Phase 3.9.3 Library Collections Mounted Management UI

Backlog: `TASK-10.9.3`

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_collections_panel.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify if needed: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify if needed: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create/modify: `Tests/UI/test_product_maturity_phase39_library_collections.py`
- Verify: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`

- [ ] **Step 1: Add mounted red tests**

Create `Tests/UI/test_product_maturity_phase39_library_collections.py`.

Use `DestinationHarness(app, "library")` and inject a fake `library_collections_service`.

Cover:

- Pressing `#library-mode-collections` mounts `#library-collections-panel`.
- Empty state copy explains Collections without claiming sync or RAG execution.
- Creating a collection through `#library-collection-name-input` and `#library-create-collection` adds it to the list and selects it.
- Renaming through `#library-collection-name-input` and `#library-rename-collection` updates detail copy.
- Deleting through `#library-delete-collection` removes it and returns to empty state.
- A service exception renders `#library-collections-error` with retry/recovery copy.
- `#library-rag-run-query` is not mounted in Collections mode.

Expected before implementation: missing selectors.

- [ ] **Step 2: Create Collections panel widget**

Create `LibraryCollectionsPanel` with stable selectors:

- `#library-collections-panel`
- `#library-collections-empty`
- `#library-collections-error`
- `#library-collections-list`
- `#library-collection-detail`
- `#library-collection-name-input`
- `#library-collection-description-input`
- `#library-create-collection`
- `#library-rename-collection`
- `#library-delete-collection`
- `#library-collection-sync-status`
- `#library-collection-item-count`

The widget should receive a `LibraryCollectionsPanelState` and render only. Keep service mutation in `LibraryScreen`.

- [ ] **Step 3: Wire Library screen state and refresh**

In `LibraryScreen.__init__`, add:

- `_library_collections_loaded`
- `_library_collections_records`
- `_library_collections_selected_id`
- `_library_collections_error`

Add worker methods:

- `_refresh_library_collections_snapshot()`
- `_list_library_collections_snapshot()`
- `_apply_library_collections_snapshot()`

Use the existing `_resolve_maybe_awaitable()` helper so fake async and sync services both work in tests.

- [ ] **Step 4: Mount Collections panel in the existing shell**

When `_active_mode == "collections"`:

- Mount `LibraryCollectionsPanel` in `#library-source-detail` after `#library-active-mode-next-action`.
- Mount lightweight inspector status in `#library-source-inspector` if the panel does not already show it.
- Remove Search/RAG widgets when leaving `search` mode.
- Do not recompose the whole Library shell unnecessarily if targeted widget refresh is enough.

Recommended helper:

```python
async def _sync_collections_panel(self) -> None:
    await self._remove_mode_widgets("#library-collections-panel, #library-collections-inspector")
    if self._active_mode != "collections":
        return
    await self._refresh_library_collections_snapshot()
```

- [ ] **Step 5: Add create, rename, and delete handlers**

Handlers:

- `@on(Button.Pressed, "#library-create-collection")`
- `@on(Button.Pressed, "#library-rename-collection")`
- `@on(Button.Pressed, "#library-delete-collection")`
- Optional selection handler for collection row buttons, if rows are rendered as buttons.

Rules:

- Notify and keep form input visible on invalid names.
- Refresh panel state after success.
- Use service methods only, not direct DB calls.
- Keep sync copy as status text only.

- [ ] **Step 6: Add TCSS only if needed**

If new classes are required, edit `tldw_chatbook/css/components/_agentic_terminal.tcss`.

Then run:

```bash
$PY tldw_chatbook/css/build_css.py
```

If the build emits the pre-existing missing `features/_evaluation_v2.tcss` warning and exits 0, record that in implementation notes.

- [ ] **Step 7: Verify Task 3**

```bash
$PY -m pytest -q \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_product_maturity_phase3_library_contract_layout.py \
  --tb=short

git diff --check
```

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  tldw_chatbook/Widgets/Library/__init__.py \
  tldw_chatbook/Widgets/Library/library_collections_panel.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Mount Library Collections management UI"
```

Only stage CSS files if they actually changed.

---

### Task 4: Phase 3.9.4 Library Collections QA Closeout And Tracking

Backlog: `TASK-10.9.4`

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-phase-3-9-library-collections.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Modify: `backlog/tasks/task-10.9 - Product-Maturity-Phase-3.9-Library-Collections-IA-Split.md`
- Modify: `backlog/tasks/task-10.9.4 - Phase-3.9.4-Library-Collections-QA-closeout-and-tracking.md`

- [ ] **Step 1: Add tracking red test**

Update or add a tracking test asserting:

- `Docs/superpowers/trackers/product-maturity-roadmap.md` lists Phase 3.9 / TASK-10.9.
- Phase 3.9 evidence path is included.
- Roadmap residual risks still include Workspaces, Import/Export depth, server sync, and deeper Study/Search/RAG flows.
- The Phase 3 spec/roadmap mention citations/snippets as later-stage Library/Search/RAG work.

- [ ] **Step 2: Create QA evidence document**

Create `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-phase-3-9-library-collections.md` with:

- Scope.
- First-time user discovery check.
- Power-user create/select/rename/delete workflow.
- Watchlists continuity check.
- Sync honesty check.
- Verification commands.
- Functional defects.
- UX defects.
- Visual/UI defects.
- Residual risks.

Use this result language only after verification:

```text
Pass for Phase 3.9 only if Library Collections management and Watchlists continuity are both usable in the mounted app.
```

- [ ] **Step 3: Update roadmap and QA index**

Update:

- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`

Roadmap status should say Phase 3.9 verified only after the tests and QA walkthrough pass.

- [ ] **Step 4: Update Backlog via CLI**

Use Backlog CLI from the implementation worktree:

```bash
backlog task edit TASK-10.9 --plan "1. TASK-10.9.1: Watchlists IA split and compatibility labels.\n2. TASK-10.9.2: Library Collections display-state and local service contracts.\n3. TASK-10.9.3: Library Collections mounted management UI.\n4. TASK-10.9.4: QA closeout and tracking.\n\nPrimary implementation plan: Docs/superpowers/plans/2026-05-08-phase-3-9-library-collections-ia-split.md."
```

At closeout, check ACs and set done only after verification:

```bash
backlog task edit TASK-10.9 --status Done
backlog task edit TASK-10.9.4 --status Done
```

Do not mark tasks done before ACs, implementation notes, and QA evidence are complete.

- [ ] **Step 5: Run focused verification**

```bash
$PY -m pytest -q \
  Tests/Library/test_library_collections_state.py \
  Tests/Library/test_library_collections_service.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_shell_destinations.py \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/Home/test_active_work_adapter.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_product_maturity_phase3_layout_contracts.py \
  --tb=short

git diff --check
```

- [ ] **Step 6: Manual QA walkthrough**

Run the app from a clean temporary HOME/XDG if feasible:

```bash
HOME=/tmp/tldw-chatbook-phase39-home XDG_CONFIG_HOME=/tmp/tldw-chatbook-phase39-config XDG_DATA_HOME=/tmp/tldw-chatbook-phase39-data $PY -m tldw_chatbook.app
```

Walk through:

- Home -> Library -> Collections discovery.
- Create collection.
- Select collection.
- Rename collection.
- Delete collection.
- Confirm sync is local-only/unavailable, not an enabled fake action.
- Open Watchlists and confirm it is Watchlists-only.
- Confirm Watchlists active-run follow-through still works in fixture-backed tests.

If a clean runtime cannot launch due local optional-service blockers, record the blocker and mounted-test evidence. Do not claim full manual pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  Tests/UI/test_product_maturity_phase3_layout_contracts.py \
  Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-phase-3-9-library-collections.md \
  Docs/superpowers/qa/product-maturity/phase-3/README.md \
  Docs/superpowers/trackers/product-maturity-roadmap.md \
  backlog/tasks/task-10*.md
git commit -m "Record Library Collections QA closeout"
```

---

## Execution Notes

- Execute tasks sequentially. Task 2 can be developed in parallel with Task 1 only if write scopes are isolated, but the safer path is sequential because label changes affect many tests.
- Keep commits PR-sized and reviewable.
- If implementation discovers that `TASK-10.9.2` persistence should be a different DB boundary, update the task AC before coding.
- If item membership becomes larger than a simple current-item seam, defer it to a follow-up and keep Phase 3.9 focused on local Collection management.
- After the plan is implemented and verified, open a PR against `dev`.
