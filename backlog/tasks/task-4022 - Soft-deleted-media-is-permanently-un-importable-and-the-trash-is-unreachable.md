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
only update) didn't previously touch `is_trash`/`trash_date`/`url` at all
(only the full-content-update path did, via `_media_payload()`'s
unconditional reset), so it now also writes `is_trash=0, trash_date=NULL`
and is forced to run even when nothing else changed, instead of taking the
pre-existing "already up-to-date" no-op shortcut. Both success paths now
return a "restored from trash" message. Because `add_media_with_keywords`
already dispatches post-ingest callbacks off `media_id is not None`, and
the app's own `was_duplicate = media_id is None` check already drives the
ingest row's "done"/"matched" copy, no changes were needed in `app.py` or
the ingest UI at all -- the DB-layer fix alone makes the re-import report
as a normal, successful import.

**Review round 1 correction.** The first version of this fix left the
metadata-only (content-identical) sub-path writing `is_trash`/`trash_date`
but NOT `url` -- and this is precisely the one case `is_canonicalisation`
(the pre-existing `overwrite=False`-only branch that used to run for a
non-trashed identical-content match) was written to cover, since
`is_canonicalisation` requires `content_hash == existing_hash` by
definition, which is exactly the condition that routes into the
metadata-only branch, never the full-update one. My original self-review
claimed the full-update path's unconditional `url` write made dropping
`is_canonicalisation` "a strictly stronger form of the same
canonicalization" -- **that was wrong**: the full-update path is never
reached when content is identical, so nothing was canonicalizing `url` for
a restored identical-content match. Reviewer reproduced it against a real
DB: a row created at an auto-generated `local://...` url, trashed, then
re-imported at a real path with identical bytes came back `is_trash=0`
("restored from trash") but still addressed by the STALE `local://...`
url -- `get_media_by_url(<the real path just imported>)` returned `None`
for a live, un-trashed item. Fixed by extending the metadata-only branch's
`UPDATE` to also write `url = ?` (and the matching sync-log payload) when
`restoring_from_trash`, mirroring what the full-update path already did.
New regression test: `test_reimport_identical_content_at_new_url_
canonicalizes_url` (RED confirmed via `git apply -R` before the fix, exact
reproduced symptom: `url` stays at the stale `local://...` value).

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

**Review round 1 correction (Undo focus).** `_undo_library_media_bulk_
delete`'s completion tail called `self.refresh(recompose=True)` (needed so
the rail's "Media N" count repaints) but never armed keyboard entry focus
afterward, unlike `_delete_library_media_selection`'s own tail, which does
so unconditionally specifically because `recompose=True` destroys and
remounts the focused button -- its own comment says "now armed on EVERY
completion path, not just full success (review round 2)". My original
Concerns section framed the Undo gap as full-success-only; the reviewer
correctly pointed out `recompose=True` always destroys the DOM regardless
of outcome, so a partial Undo failure (which re-renders the narrowed
receipt with a brand-new "Undo" button instance) loses focus exactly the
same way. Fixed with one `self._arm_library_list_entry_focus()` call in
the same `if self.is_mounted:` block; pinned by new assertions on both the
full-success and partial-failure real-DB Undo tests (RED confirmed via
`git apply -R` before the fix -- both failed on the missing
`_entry_focus_arm_calls == [True]` assertion).

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
(4 tests, real file-backed `MediaDatabase` -- M7 correction: this was
written as "3 tests" below when first drafted, but round 1 added a fourth,
`test_reimport_identical_content_at_new_url_canonicalizes_url`, without
this paragraph being updated to match) covers url-match restore,
hash-fallback restore, a guard rail that an ACTIVE duplicate is still
skipped exactly as before, and round 1's url-canonicalization regression.
`Tests/UI/test_library_multiselect_media.py`
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

**Review round 2 (final-review fix wave, `fix/library-recritique-p1s`):
scoped restore to opt-in and fixed three defects the unconditional form
of round 1 introduced.** A second review of this task's DB-layer hunk
found 1 Critical + 3 Important, all confined to
`_add_media_with_keywords_impl`, plus a records-hygiene item on the
critique doc itself:

- **Adopted the reviewer's architectural recommendation first:** added
  `restore_trashed: bool = False` to `add_media_with_keywords` /
  `_add_media_with_keywords_impl`. `restoring_from_trash` is now
  `bool(row["is_trash"]) and restore_trashed` instead of unconditional on
  any trashed match -- the `overwrite=False` "never mutate an existing
  row" contract is intact again for every caller that doesn't ask for
  restore. Only `Local_Ingestion/local_file_ingestion.py`'s
  `persist_parsed_media` (the real Library ingest writer, both its
  `app.py` and `ingest_local_file` call paths) passes
  `restore_trashed=True` -- it is the one caller whose user-facing
  behavior actually needs "re-importing this file un-trashes it."
  `ingest_article_to_db_new`/`import_obsidian_note_to_db` (module-level
  wrappers) were left unchanged: neither is named by any finding, and
  scoping the flag to just the flagged callers keeps the blast radius
  exactly as small as the reviewer asked for.
- **C1 (Critical) fixed:** `_persist_chunks` took a `replace_existing`
  parameter instead of reading the outer `overwrite` closure variable;
  both restore sub-paths now pass `overwrite or restoring_from_trash`, so
  a chunked re-import of a trashed row deletes the stale chunk rows
  before inserting instead of colliding on
  `UNIQUE(media_id, chunk_index, chunk_type)`. (SQLite treats a NULL
  `chunk_type` as always distinct in a UNIQUE index, so the regression
  tests had to give chunks an explicit, shared `chunk_type` to actually
  reproduce the collision -- an omitted one would have passed for the
  wrong reason.)
- **I1 (Important) resolved by construction:** the opt-in flag itself is
  the fix. Verified all three non-import callers I1 named still leave a
  trashed match untouched, since none passes `restore_trashed=True`:
  `Chatbooks/chatbook_importer.py`'s `_import_media` (real-DB regression
  test added), `Media/local_media_reading_service.py`'s
  `_materialize_reading_import_row` content-hash leg (real-DB regression
  test added), and `UI/Console_Modules/message.py`'s
  `_save_console_message_as_media` (verified by reading the call site --
  no `restore_trashed` kwarg passed; a full Textual-pilot test was judged
  disproportionate for a call site with no opt-in flag to test).
- **I2 (Important) fixed:** both restore branches now skip
  `update_keywords_for_media` when `restoring_from_trash and not
  keywords_norm` -- an empty incoming keyword list on a restore no longer
  wipes the row's existing, user-curated keywords. A restore that DOES
  supply keywords still applies them normally (unchanged, pinned by a
  guard-rail test).
- **I3 (Important) fixed:** the identical-content restore branch's `url`
  write is now gated on `restore_canonicalizes_url` (`existing_url`
  starts with `local://` and the new `url` doesn't), mirroring the
  pre-existing `is_canonicalisation` branch's one-directional rule
  instead of reversing it. A row imported from a canonical source url,
  trashed, then re-imported from a local file path with identical
  content now keeps its canonical url. Also fixed the stale "never
  touches url" test comment this drift left behind (`Tests/Media_DB/
  test_media_db_v2.py`, the hash-fallback test) and added a real
  assertion pinning it.
- **Every fix was mutation-tested**, not just written-then-run-once: each
  of C1/I1/I2/I3 was reverted in isolation via targeted `Edit` calls
  (never `git checkout`/stash, per this repo's mutation-restore lesson),
  confirmed the relevant new test(s) failed with the exact symptom the
  finding described (`sqlite3.IntegrityError` for C1, resurrection for
  I1, wiped keywords for I2, clobbered canonical url for I3), then
  restored.
- **New tests** (all real, file-backed `MediaDatabase`, no mocks):
  `TestRestoreTrashedIsOptIn` (2), `TestReimportAfterTrashChunks` (2),
  `TestReimportAfterTrashKeywords` (2),
  `TestReimportAfterTrashUrlCanonicalization` (1), and
  `TestReimportAfterTrashCombined` (1) -- the last is the reviewer's own
  suggested coverage: ONE test exercising restore with chunks, existing
  keywords, and a canonical url together, which alone would have caught
  C1+I2+I3. All in `Tests/Media_DB/test_media_db_v2.py`. Plus one
  caller-level regression test each in `Tests/Chatbooks/
  test_chatbook_importer.py` and `Tests/Media/
  test_local_media_reading_service.py` for I1(a)/I1(b). The pre-existing
  `TestReimportAfterTrash` tests were updated to pass
  `restore_trashed=True` explicitly, matching what the real ingest writer
  now does.
- **I4 (records):** fixed the re-critique document itself --
  `p1_count: 9` -> `8` in the frontmatter (the correction note already
  said to read it as 8), added an inline withdrawal marker on the RC-02
  bullet in the P1 list (68 lines below the correction, it still read as
  a live finding), and corrected the conclusion's "Two of our own shipped
  fixes (nav ghosting, blank-note GC)" to name only the blank-note GC
  (nav ghosting was a measurement artifact, not a broken fix).
- **Minors:** re-stamped `Docs/User_Guide/library/media-and-conversations.md`
  from a branch SHA (`8bb6dd730`, a commit on THIS branch, not dev) to the
  actual dev merge-base (`e13608106`); fixed this task's own "3 tests"
  claim above to "4" (M7).
- **Targeted suites, 0 regressions:** `Media_DB` (94 passed, 6 skipped --
  sync-server integration), `Chatbooks` (188 passed, 1 skipped -- slow
  opt-in), `Media/test_local_media_reading_service.py` (67 passed),
  `UI/test_library_multiselect_media.py` (38 passed),
  `Local_Ingestion`+`Library/test_library_ingest_*` (207 passed). A
  `--collect-only -q` sweep across every touched tree (2,837 tests) found
  no import errors.

**Files changed (round 2):**
- `tldw_chatbook/DB/Client_Media_DB_v2.py` -- opt-in `restore_trashed`
  flag, `_persist_chunks(replace_existing=...)`, I2 keyword-skip, I3 url
  gating
- `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` --
  `persist_parsed_media` passes `restore_trashed=True`
- `Tests/Media_DB/test_media_db_v2.py` -- 4 existing tests updated
  (`restore_trashed=True`), 8 new tests across 5 new classes
- `Tests/Chatbooks/test_chatbook_importer.py`,
  `Tests/Media/test_local_media_reading_service.py` -- one caller-level
  regression test each
- `Docs/User_Guide/library/media-and-conversations.md` -- SHA re-stamp
- `.impeccable/critique/2026-08-09T20-15-07Z__tldw-chatbook-ui-screens-library-screen-py.md`
  -- I4 corrections
- `backlog/tasks/task-4023 - *.md` -- cross-task reversibility-inconsistency
  note

**Files changed (round 1):**
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
