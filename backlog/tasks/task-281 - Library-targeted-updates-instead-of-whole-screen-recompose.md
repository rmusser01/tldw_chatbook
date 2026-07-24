---
id: TASK-281
title: Library: targeted sync_state updates instead of 124 whole-screen recomposes
status: In Progress
assignee: ['@claude']
created_date: '2026-07-16 14:30'
updated_date: '2026-07-16 21:20'
labels: [performance, library]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
library_screen.py calls self.refresh(recompose=True) — full remove/remount of nav, footer, ~20-row rail, and 50-100-row canvas — from 124 sites including per-row checkbox handlers. LibraryRail/LibraryMediaCanvas/LibraryConversationsCanvas.sync_state() exist for targeted updates and have ZERO callers. This exact pattern caused the app-wide mouse-capture bug base_app_screen.py works around. Stage by interaction class (checkbox toggles first). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Per-row selection/checkbox interactions no longer recompose the screen (targeted row/canvas updates; verified by interaction tests)
- [x] #2 Rail counts stay consistent with canvas state across the converted interactions
- [x] #3 Remaining recompose sites inventoried with a justification or follow-up
- [x] #4 Live QA on the converted flows
<!-- AC:END -->

## Implementation Plan

Staged to the SELECTION interaction class only (per the decided two-tier
design in `.superpowers/sdd/task-252-brief.md`):

1. Tier 1 (in-place, no rebuild): a shared `_apply_library_row_toggle(screen,
   kind, button, row_id)` helper flips a select-mode checkbox row's marker,
   the "N selected" Static, and export-selected's disabled state, for the
   conversations/media/notes row-press handlers' select-mode branches.
2. Tier 2 (canvas-scoped recompose): a shared `_sync_library_canvas(screen,
   kind)` helper calls the mounted canvas's own `sync_state(new_state)` —
   releasing `App.mouse_captured` first, mirroring
   `BaseAppScreen.refresh`'s guard — for browse-mode row selection and
   select-mode enter/exit/select-all/clear.
3. Both helpers fall back to the old `screen.refresh(recompose=True)` on
   any failure (never crash), and are written as MODULE-LEVEL functions
   (not `LibraryScreen` methods) for compatibility with existing
   bare-fake unit tests.
4. New RED-first test file `Tests/UI/test_library_selection_updates.py`;
   keep the full Library + destination-shell suites green.
5. Grep remaining `self.refresh(recompose=True)` sites and append an
   AC #3 inventory below.

## Implementation Notes

**Approach.** Added two module-level helpers directly above `class
LibraryScreen` in `library_screen.py` (~line 504):
`_apply_library_row_toggle(screen, kind, button, row_id)` (Tier 1) and
`_sync_library_canvas(screen, kind)` (Tier 2). Both take the screen
instance as an explicit first argument rather than being `LibraryScreen`
methods — `Tests/UI/test_library_multiselect_{conversations,media,notes}.py`
drive several handlers directly against a bare `SimpleNamespace` fake that
only stubs `.refresh`; a `self.<new_method>(...)` call would raise
`AttributeError` at the attribute-lookup step (before the method's own
`try/except` could run) since the fake has no such attribute, breaking
those existing (read-only) tests. A module-level function has no such
lookup — `screen.query_one(...)` etc. fail *inside* the function and are
caught by its own `except`, falling through to the already-stubbed
`screen.refresh(...)`, exactly matching those tests' existing
expectations. Verified empirically: all 13 existing multiselect tests
plus the full `test_library_shell.py` (257 tests) and
`test_destination_shells.py` (103 tests) stayed green with no edits.

**Converted (10 call sites / 9 handler functions):**
- Tier 1: `handle_library_conversation_row`, `handle_library_media_row`,
  `handle_library_notes_row` — each handler's select-mode branch.
- Tier 2: `handle_library_conversation_row`'s browse-mode branch (the
  `▸` preview-select interaction); `handle_library_conversations_select_toggle`,
  `_select_all`, `_select_clear`; `handle_library_media_select_toggle`,
  `_select_all`, `_select_clear`.

**Deviation from the brief, flagged per its own STOP-on-conflict
guardrail:** the brief's Tier 2 section assumed all three canvases have a
`sync_state()` hook ("call the CANVAS's existing `sync_state`... Same for
[notes'] select-mode enter/exit toggle, select-all, and clear"). They
don't: `LibraryNotesCanvas.__init__` has its own `sync_state` *parameter*
(the unrelated notes-sync-panel display state, `LibraryNotesSyncState`),
which shadows the method name a targeted-update hook would need — the
audit doc itself (§P1 B2) only lists `LibraryRail`, `LibraryMediaCanvas`,
and `LibraryConversationsCanvas` as having `sync_state()`, correctly
omitting notes. `handle_library_notes_select_toggle` / `_select_all` /
`_select_clear` and notes' non-select-mode row press (opens the in-canvas
editor, a structural mode swap + async detail fetch) were left as full
`self.refresh(recompose=True)`, each with a docstring note pointing here.
Similarly, `handle_library_media_row`'s non-select-mode branch
(`_open_library_media_viewer`) was left unconverted: it swaps the
mounted widget class entirely (`LibraryMediaCanvas` → `LibraryMediaViewer`,
a different class with no `sync_state`) and kicks an async detail-fetch
worker, so it doesn't fit either tier's shape.

**AC #3 inventory** (grepped after the change; 121 total original
`self.refresh(recompose=True)` call sites in `library_screen.py`, 2
additional new *defensive-only* fallback sites added inside the two
helpers themselves, never exercised on the happy path):
- **(a) Converted this task:** 10 sites / 9 handler functions (listed
  above).
- **(b) Structural/data-mutation — full recompose stays the conservative
  correct choice:** 107 remaining sites. Categories: delete flows
  (`_delete_library_note/_media_item/_prompt/_skill`), save/conflict
  flows (`_save_library_note`, `_resolve_library_{note,prompt}_conflict`,
  `handle_library_media_{analysis,edit}_save`), async detail-fetch
  workers landing data (`_refresh_library_{note,media,prompt,skill}_detail`),
  rail-changing navigation (`_select_library_rail_row`,
  `_open_library_item_by_id`), mode/view swaps (notes editor/sync back,
  prompts/skills editor back, media viewer back), sort/filter cycling
  (notes/prompts/skills sort and filter, RAG scope toggles), ingest job
  lifecycle (retry/dismiss/clear-finished/analyze-toggle/chunk-toggle),
  import/export submission and destination-browse callbacks, and
  `on_mount`/`create_local_workspace`. All either mutate persisted data,
  change what the RAIL shows, or swap the mounted widget class — outside
  this task's SELECTION-interaction-class scope.
- **(c) Candidates for future conversion:** 4 sites.
  `handle_library_notes_select_toggle` (line ~10471),
  `_select_all` (~10483), `_select_clear` (~10494) — blocked by the
  `LibraryNotesCanvas.sync_state` name collision above; unblocking needs
  `LibraryNotesCanvas`'s constructor `sync_state` kwarg renamed (e.g. to
  `notes_sync_state`) and a real `sync_state(list_state)` method added,
  then wiring `kind == "notes"` into the existing `_sync_library_canvas`.
  `handle_library_media_type_filter_pressed` (~6144) — cycles the media
  type filter and clears select mode/selection, entirely within the
  media canvas (no rail change); could route through the already-existing
  `_sync_library_canvas(self, "media")` today, in a follow-up.

**Tests.** New `Tests/UI/test_library_selection_updates.py` (5 tests, all
written RED-first against the pre-fix code): checkbox toggle skips a
screen recompose (confirmed RED pre-fix: recompose count rose by 1);
rail row survives a toggle by object identity (RED pre-fix: a fresh
`LibraryRail` is minted on every full recompose); browse-mode row
selection routes through `LibraryConversationsCanvas.sync_state` and not
a screen recompose (RED pre-fix: zero `sync_state` calls); Tier 2
releases `App.mouse_captured` before the canvas sync (passed pre-fix too,
by coincidence — `BaseAppScreen.refresh`'s own guard covered it via the
old code path; the manual release now covers the same case once that
path is bypassed); forcing `query_one` to raise inside the Tier 1 helper
falls back to a full recompose without crashing, and the selection state
still reflects the toggle.

**Files changed:**
- `tldw_chatbook/UI/Screens/library_screen.py` — added
  `_apply_library_row_toggle` / `_sync_library_canvas`; rewired the 10
  converted call sites; added docstring notes on the 3 notes select-mode
  handlers explaining why they're unconverted.
- `Tests/UI/test_library_selection_updates.py` (new).

**Not done (by design):** AC #4 (live QA on the converted flows) is
explicitly out of scope for this pass — left for the controller.
