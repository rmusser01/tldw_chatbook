# Database Notes Folder Navigator Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Replace the flat Database Notes list with a lazy, placement-aware folder navigator that supports manual organization without regressing the existing note editor or File Notes.

**Architecture:** A pure projection module will merge bounded `NoteFolderPage` batches into immutable tree rows and stable navigation state. `LibraryScreen` will own service calls, generations, mutations, and focus restoration; `LibraryNotesCanvas` will only render the projection and emit controls. All storage access remains behind `NotesScopeService`, in accordance with ADR-059 and ADR-073.

**Tech Stack:** Python 3.11+, Textual 8.x, dataclasses, pytest/pytest-asyncio, Rich/Textual screenshot export.

**Backlog task:** `TASK-15706`

**Architecture decisions:**

- ADR required: no
- Existing ADRs: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`; `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
- Reason: these ADRs already define normalized folder services, placement identity, managed ownership, lazy rendering, and local/server separation.

**Known baseline:** The focused baseline has 109 passing tests and one unrelated existing failure in `Tests/UI/test_library_multiselect_notes.py`; its fake screen omits the already-required `_library_notes_mutation_in_flight` attribute.

---

## Task 1: Pure placement-aware tree projection

**Files:**

- Create: `tldw_chatbook/Library/library_notes_tree_state.py`
- Create: `Tests/Library/test_library_notes_tree_state.py`

1. Write failing tests for root/nested ordering, Unfiled, distinct placement IDs for one note, breadcrumb derivation, manual ancestor+descendant duplicates, generated managed-ancestor collapse, and inactive-owner decoration.
2. Run `python -m pytest Tests/Library/test_library_notes_tree_state.py -q` and confirm the missing module/behavior fails.
3. Add immutable row, projection, cache, paging, and navigation dataclasses. Keep folder placement identity separate from note identity, use semantic glyph+label text in addition to style roles, and expose bounded-load cursors.
4. Implement pure page merging and visible-row projection. Preserve all surviving expanded IDs, prefer the same placement on refresh, and fall back to another placement of the same note only when needed.
5. Re-run the focused tests to green, then run Ruff on the new module and test.

## Task 2: Render the lazy folder navigator

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`
- Modify: `tldw_chatbook/css/components/_library.tcss` or the existing Library stylesheet owning Notes geometry

1. Add failing widget tests for folder expand controls, nested indentation, Unfiled, duplicate note placement metadata, breadcrumbs, non-color managed/inactive-managed cues, loading/more rows, and compact labels.
2. Extend the canvas list inputs with an optional tree projection and compose folder/note buttons using stable DOM-safe IDs. Continue assigning `note_id` for the existing editor/select handler and also assign `placement_id`, `folder_id`, `membership_id`, `breadcrumb`, `ownership`, and `owner_active`.
3. Add keyboard-operable action buttons and semantic classes using theme tokens only. Ensure glyphs and text remain sufficient with color removed.
4. Verify widget tests and export a 60x20 frame for inspection.

## Task 3: Load and preserve tree state through LibraryScreen

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Create: `Tests/UI/test_library_notes_folder_navigator.py`

1. Add failing screen tests proving the initial root batch is loaded once, expansion issues one bounded bulk request, no per-note detail calls occur until a note is opened, stale generations are ignored, and in-memory async tests stay on their owning thread.
2. Add screen-owned tree cache, expanded IDs, paging state, placement focus, loading/error state, and request generation. Start the root load on the Database Notes route and merge expanded-folder batches through the pure projector.
3. Use the normalized `NotesScopeService` methods only. Route file-backed synchronous calls through the existing worker boundary and allow already-async/in-memory test doubles to remain thread-local.
4. Extend semantic focus capture/restore to use placement identity first and note identity second; keep the active editor keyed only by note ID.
5. Run the focused screen and current Notes workflow tests.

## Task 4: Manual folder and placement operations

**Files:**

- Create: `tldw_chatbook/Widgets/Library/library_note_folder_dialog.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_notes_folder_navigator.py`

1. Add failing tests for create, rename, move, remove, restore, attach, detach/move placement, optimistic conflicts, cancellation, and managed/inactive-managed protection.
2. Add a small reusable modal for folder name/target selection and explicit confirmation for remove. Keep restored folders discoverable through a visible recovery action.
3. Wire mutations through `NotesScopeService`, supplying expected versions and membership versions where required. Treat moving a manual note as attach-then-detach with safe recovery if detach fails; adding a placement never removes existing memberships.
4. Disable manual mutation controls for managed or inactive managed rows and explain why in visible text/tooltips.
5. Reload affected branches and restore expansion/focus/editor identity after each successful mutation; surface collision, conflict, capability, and generic failures without discarding state.

## Task 5: Selection, filtering, responsive layout, and regressions

**Files:**

- Modify: `tldw_chatbook/Library/library_notes_tree_state.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_notes_folder_navigator.py`
- Modify: `Tests/UI/test_library_multiselect_notes.py` only for the confirmed baseline fixture omission
- Modify: relevant rendered/accessibility Library tests under `Tests/UI/`

1. Add failing tests for duplicate-placement selection, filter breadcrumbs, reorder/refresh preservation, collapsed-folder fallback, resize/recompose focus, and exact 60x20 allocation.
2. Make selection note-based across duplicate placements while focus remains placement-based. Filter by note title/path and show every matching breadcrumb without mutating expansion state.
3. Keep navigator action rows bounded at 60 columns and preserve the existing editor, create, sync, and File Notes routes.
4. Add a rendered-frame assertion/export and inspect hierarchy, status, controls, and clipping with theme color ignored.

## Task 6: Performance, host routing, and verification

**Files:**

- Modify: `Tests/UI/test_library_notes_folder_navigator.py`
- Modify: `Tests/UI/test_library_shell.py` or the narrow existing host-routing test
- Modify: `backlog/tasks/task-15706 - Render-and-operate-the-Database-Notes-folder-navigator.md`

1. Add request-count tests proving bulk loads are bounded and no N+1 note calls occur for a representative multi-folder tree.
2. Run focused suites for tree state, canvas, screen, folder service, current note editor, multiselect, responsive Library, and File Notes.
3. Run the broader Library/Notes test groups plus Ruff on changed Python files.
4. Exercise a real file-backed database and restart the app against a scratch profile; verify root load, nested expansion, duplicate placement opening, folder mutation, editor return, File Notes routing, and a 60x20 frame.
5. Self-review the diff, record evidence and the known-baseline disposition, add concise Implementation Notes with ADR links, check every acceptance/DoD box, and set the task status to Done only if every requirement has evidence.
