---
id: TASK-4022
title: Soft-deleted media is permanently un-importable and the trash is unreachable
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 20:30'
updated_date: '2026-08-09 22:58'
labels:
  - library
  - media
  - data-loss
  - recritique-2026-08-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09 (RC-04/RC-05), reproduced by the mechanical arm at dev `4d0232358`.

Repro: import a file → Media ▸ Select → check it → Delete selected → confirm. Then re-import the
same file. Result:

    ≡ matched · short.txt
    Already in Library — matched an existing item; nothing new was imported.

…while `Media (1)` and the item is absent from every list. The import dedup matches
**soft-deleted** rows, so a deleted file can never be re-added. Meanwhile the confirmation dialog
promises `This moves them to trash.` and there is no trash anywhere in the product — not in the
rail, not in the `type:` filter (which offers only `All` and the ingested types), not on any canvas.

Net effect: the user's content is neither present nor restorable through the UI, and the one action
that promised reversibility is the one that makes it unreachable.

Two coupled defects, both in scope here:
1. Dedup must not match soft-deleted rows (or must offer to restore the existing row instead of
   silently refusing the import).
2. Bulk delete completes with no receipt and no undo. Compare the asymmetry: creating one item
   yields `✓ done · file · 1s` plus an `Open in Library` jump; destroying two yields silence.

Either ship the trash the copy promises (a `type: Trash` value or a rail row, with restore), or
change the copy to state what actually happens — but the current combination of a reversibility
promise, no destination, and a permanent import block is the worst of the three options.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A file deleted from Media can be re-imported, or the duplicate-match path offers restore instead of silently refusing
- [x] #2 Bulk delete emits a receipt naming the count, with an undo affordance at the point of action
- [x] #3 The confirmation copy and the product agree: either the trash is reachable and restores, or the copy stops promising it
- [x] #4 Live verification of the full cycle: import → delete → re-import → the item is present exactly once
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both defects at HEAD against a real (file-backed) DB: import short.txt, mark it as trash via mark_as_trash, confirm get_media_by_url/get_media_by_hash already exclude it (they do -- default include_trash=False), then trace the ACTUAL dedup path add_media_with_keywords uses during ingest (persist_parsed_media -> _add_media_with_keywords_impl's own inline SELECT), which filters only 'deleted = 0' and NOT 'is_trash = 0' -- root cause of defect 1.
2. Root-cause the receipt gap: _delete_library_media_selection (bulk) and _delete_library_media_item (single) both already call mark_as_trash via media_reading_scope_service.delete_media_item -- correct soft-delete seam, no raw SQL -- but neither path emits any success notification/receipt; only failures notify.
3. Decide dedup behavior: prefer match-and-restore over exclude-deleted (content still exists, restoring is what the user wants), gated on verifying the restore is sound -- confirm _media_payload() already unconditionally resets is_trash/trash_date/deleted on the full-update path, and extend the metadata-only path to do the same.
4. TDD RED first for both defects (revert-with-patch-and-restore method, not stash) reproducing the OBSERVED effect: media_id=None + 'already exists. Overwrite not enabled.' message for defect 1; AttributeError for missing receipt/Undo/Dismiss handlers and fields for defect 2.
5. Implement: DB-layer restoring_from_trash branch in _add_media_with_keywords_impl (Client_Media_DB_v2.py); a delete_receipt_count field threaded through LibraryMediaCanvasState/build_library_media_state; a _library_media_delete_receipt_ids screen field set on delete completion, cleared on new-confirm-arm/fresh-select-entry; Undo (calls restore_media_item via the same scope-service seam) and Dismiss handlers; canvas rendering of the receipt row; honesty-fix the bulk AND single-item delete confirm copy since both promised a Trash view that doesn't exist.
6. Verify RED->GREEN via revert-patch-restore for both defects; run targeted real-DB test suites (Media_DB, Library, UI multiselect); live-verify the full import->delete->re-import cycle plus Undo in tmux.
7. Decide the Trash-view question: conclude a persistent browsable Trash surface is out of scope for this task (the acute defects are fixed without it); make the confirm copy honest about what actually ships; file it separately (scanned IDs fresh, landed on task-4025 after the CLI's auto-assignment collided with an existing task-13213 on origin/dev).
8. Backlog hygiene, commit, self-review, report.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Dedup decision: restore-on-match, not exclude-deleted.** The row a trashed
match points to still carries its full content, so refusing to touch it and
just falling through to a fresh INSERT would either violate the `url`/
`content_hash` UNIQUE constraints or fork the same logical item into two
rows. Restoring the existing row is both the technically sound option and
the one that matches user intent ("I want this file in my library again").

**Root cause (defect 1).** `get_media_by_hash`/`get_media_by_url` already
excluded trashed rows by default (`include_trash=False`) -- they were never
the bug. The actual dedup decision lives in `_add_media_with_keywords_impl`'s
own inline SQL (`Client_Media_DB_v2.py`), which filtered `deleted = 0` but
never `is_trash = 0`. A trashed row therefore still matched, and with
`overwrite=False` (what the real ingest writer always passes) that hit the
"already exists, do nothing" branch, returning `media_id=None` -- the row
stayed trashed forever and the ingest job reported `Already in Library`
with no way back.

**Fix.** The initial url/hash lookup now also selects `is_trash`. A matched
row with `is_trash=1` is now routed through the SAME full-update code path
`overwrite=True` uses (`if overwrite or restoring_from_trash:`), regardless
of the caller's `overwrite` flag -- a trashed row isn't an active duplicate
to protect from being clobbered. The content-identical sub-path (metadata-
only update) didn't previously touch `is_trash`/`trash_date` at all (only
the full-content-update path did, via `_media_payload()`'s unconditional
reset), so it now also writes `is_trash=0, trash_date=NULL` and is forced
to run even when nothing else changed, instead of taking the pre-existing
"already up-to-date" no-op shortcut. Both success paths now return a
"restored from trash" message. Because `add_media_with_keywords` already
dispatches post-ingest callbacks off `media_id is not None`, and the app's
own `was_duplicate = media_id is None` check already drives the ingest
row's "done"/"matched" copy, no changes were needed in `app.py` or the
ingest UI at all -- the DB-layer fix alone makes the re-import report as a
normal, successful import.

**Defect 2 (no receipt/undo).** Both `_delete_library_media_selection`
(bulk) and `_delete_library_media_item` (single) already went through the
correct soft-delete seam (`media_reading_scope_service.delete_media_item`
-> `MediaDatabase.mark_as_trash`, never raw SQL) -- the gap was purely that
neither emitted anything on success. Added `LibraryMediaCanvasState.
delete_receipt_count` (pure passthrough, mirrors `confirming_bulk_delete`)
and a screen-side `_library_media_delete_receipt_ids` tuple, set to the
succeeded subset when a bulk delete completes, cleared when a new bulk-
delete confirmation is armed or Select mode is freshly re-entered. The
canvas renders a `✓ deleted · N items` row (reusing the already-proven-safe
`library-toolbar-count` class, per the task-2853 unbounded-Static lesson
recorded in this same file) with `Undo`/`Dismiss` buttons, positioned
outside `select_mode` since a full-success delete exits it. `Undo` calls
`media_reading_scope_service.restore_media_item` (mode="local") -- the
scope-service method already existed, fully wired to `MediaDatabase.
restore_from_trash`, and was simply never called from any UI path before
this. It returns the freshly restored row, which is inserted straight back
into `_local_source_records["media"]` (deduped by `_source_record_id`
against what's already cached, so a stale receipt's Undo on an item
restored some other way -- e.g. by re-importing -- is a safe no-op, not a
duplicate row; verified live). A partial Undo failure narrows the receipt
to just the still-failed ids rather than clearing it, mirroring the
delete path's own partial-failure behavior.

**AC#3 / the Trash-view question.** Concluded a persistent, browsable Trash
surface (a rail entry, a `type: Trash` filter value, or a dedicated canvas)
is its own task, not a one-liner alongside the two acute defects here --
filed separately as **task-4025** (ID scanned fresh: the CLI's own
auto-assignment landed on `task-13213`, which already exists on
`origin/dev` under an unrelated title -- confirms the standing lesson about
never trusting the CLI's auto-numbering; renumbered by hand to the verified-
free `task-4025` in the 40xx range this programme uses). In this task, both
delete confirmations (`Widgets/Library/library_media_canvas.py`'s bulk copy
and `Widgets/Library/library_media_viewer.py`'s single-item copy) were
rewritten to stop promising a Trash view that doesn't exist: the bulk copy
now says "You can undo right away — there's no Trash view to browse
later.", and the single-item copy (which has no in-place Undo) says
"Re-import the same file later to bring it back — there's no Trash view to
browse."

**Tests.** New: `Tests/Media_DB/test_media_db_v2.py::TestReimportAfterTrash`
(3 tests, real file-backed `MediaDatabase`) covers url-match restore,
hash-fallback restore, and a guard rail that an ACTIVE duplicate is still
skipped exactly as before. `Tests/UI/test_library_multiselect_media.py`
gained 3 real-DB Undo tests (full success, partial failure, a
duplicate-insert guard) plus 4 handler-level tests (Undo dispatch/no-op/
in-flight-guard, Dismiss) and 2 canvas-render tests for the receipt row;
the 3 existing real-DB delete tests were extended with receipt assertions.
`Tests/Library/test_library_media_state.py` gained a passthrough test for
`delete_receipt_count`. RED was captured for BOTH defects by reverting each
production diff via `git apply -R` (never stash), confirming the tests fail
with the exact observed symptoms (`media_id=None` + "already exists.
Overwrite not enabled." for defect 1; `AttributeError`s for the missing
receipt/Undo/Dismiss surface for defect 2), then restoring and re-running
green. Targeted suites (`Media_DB`, `Library`, `Media`, the touched `UI`
file) are 0 regressions; a `--collect-only -q` sweep across all touched
trees found no import errors; the one failure seen in a broader `Tests/
Library` run (`test_shadow_name_set_stays_in_sync_with_real_sources`,
video-gen shadow names) is the pre-existing ambient failure this
programme's own Global Constraints section names -- unrelated to this
task, not fixed here.

**Live verification (task-4022 AC#4).** tmux socket `rcT3lib6158`, scratch
`/tmp/rcT3`, `users_name = sdd_rct3`. Full cycle: imported `short.txt`
(`✓ done · short.txt · 1s`, `Media (1)`) -> Select -> checked it -> Delete
selected (confirm copy read exactly "Delete 1 selected item? You can undo
right away — there's no Trash view to browse later.") -> confirmed
(`Media (0)`, receipt row `✓ deleted · 1 item   Undo   Dismiss` rendered)
-> re-imported the identical path -> pre-flight correctly forecast "1 will
import" (not a false "already in library"), and the run completed as
`✓ done · short.txt · 1s` / `Imported short.txt` / `Open in Library`,
NEVER `matched` -> `Media (1)`, exactly one row, confirmed by opening the
Media list. As a bonus check, clicked the now-stale `Undo` from the earlier
receipt afterward: it no-opped safely (no duplicate row, `Media` stayed at
1), confirming the dedup guard in `_undo_library_media_bulk_delete` holds
even when a receipt outlives the item it names.

**Files changed:**
- `tldw_chatbook/DB/Client_Media_DB_v2.py` -- the dedup/restore fix
- `tldw_chatbook/Library/library_media_state.py` -- `delete_receipt_count`
- `tldw_chatbook/UI/Screens/library_screen.py` -- receipt state, Undo/
  Dismiss handlers, `_undo_library_media_bulk_delete`
- `tldw_chatbook/Widgets/Library/library_media_canvas.py` -- receipt row,
  honest bulk confirm copy
- `tldw_chatbook/Widgets/Library/library_media_viewer.py` -- honest
  single-item confirm copy
- `Tests/Media_DB/test_media_db_v2.py`, `Tests/UI/
  test_library_multiselect_media.py`, `Tests/Library/
  test_library_media_state.py` -- new/extended tests
- `Docs/User_Guide/library/media-and-conversations.md` -- updated copy,
  new receipt/Undo behavior, re-stamped
- `backlog/tasks/task-4025 - *.md` -- filed the Trash-view follow-up
<!-- SECTION:NOTES:END -->
