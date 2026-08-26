---
id: TASK-4025
title: >-
  Library media Trash view is unreachable — build a browsable, restorable
  surface
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 22:56'
updated_date: '2026-08-11 12:51'
labels:
  - library
  - media
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-4022 fixed the two acute defects (a deleted file could never be re-imported, and bulk delete had no receipt or undo) but deliberately scoped out a persistent, browsable Trash surface: today, an item moved to trash (mark_as_trash/is_trash=1) has no rail entry, no type: filter value, and no canvas anywhere it can be listed or restored from once its at-point-of-action Undo receipt is dismissed or the session ends. The only way back at that point is re-importing the exact same file (now honest and functional, but not available for content that isn't a re-importable file, and not discoverable for a user who doesn't remember what they deleted). This task is to design and ship that surface: a place to see everything currently in trash and restore it, using the existing DB-layer restore_from_trash/MediaDatabase.restore_from_trash and Media/local_media_reading_service.py's already-implemented restore_media_item (currently unwired to any UI).

Note for the design: `MediaDatabase.mark_as_trash`/`restore_from_trash` both explicitly document "does not affect FTS" -- a trashed item's content stays fully indexed in the FTS5 table and therefore still surfaces from full-text search today. This is pre-existing and symmetric (the same is true in both directions), not something task-4022 introduced or changed, but a Trash surface will make trashed-vs-active a visible distinction to the user for the first time -- this task should explicitly decide whether search results need to say/filter on trashed state, rather than leaving that decision implicit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Trash surface exists somewhere reachable from the Media canvas (rail entry, type: filter value, or dedicated canvas) listing every item with is_trash=1
- [x] #2 Each trashed item can be restored from that surface via the existing restore_from_trash/restore_media_item seam, with the list and rail counts updating in place
- [x] #3 The Media delete confirmation copy (bulk and single-item) is updated to point at the new surface instead of describing an Undo-only/re-import-only recovery path
- [x] #4 Live verification of the full cycle: delete an item, dismiss or lose its Undo receipt, find and restore it from the new Trash surface
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. **Mechanism (AC#1): a third `_library_media_view` value, `"trash"`, inside the
   Media canvas — entered via a "Trash" toolbar button on the media list, exited
   via "‹ Media"/Escape.** Judged against the alternatives:
   - *`type:` filter value*: rejected. `type:` is a CONTENT-type cycler whose
     option set is derived from the records' distinct `media_type` values
     (`build_library_media_state`); Trash is a STATE. Injecting it would lie in
     the cycle tooltip ("Cycles media type: …"), corrupt the "N of M · type: X"
     status grammar, and collide with task-14902's cycler convergence (which
     treats these as value cyclers over one dimension).
   - *Rail entry*: rejected. The rail's Browse section lists content stores
     (Media/Chats/Notes/Prompts/Skills/Collections/Search); Trash is media-only
     state, and per ADR-055's inventory notes/prompts/collections deletes have
     no trash surface (tasks 15100–15102) — a rail row would promote a
     media-scoped state to a sibling of the stores and imply a product-wide
     Trash that does not exist.
   - *In-canvas view* (chosen): the media canvas already swaps views within one
     canvas (`"list"`/`"viewer"`); Trash joins as a third view of the same
     canvas — reachable exactly where deletion happens, no new grammar.
2. **Pure state** (`Library/library_media_state.py`): `LibraryMediaTrashRow` /
   `LibraryMediaTrashState` + `build_library_media_trash_state` (auto-select
   first row, honest empty copy, truncation status). TDD first.
3. **Widget** (`Widgets/Library/library_media_trash_canvas.py`): heading
   ("‹ Media" + "Trash (N)"), status line, row buttons (`▸ ` selected marker,
   own `library-media-trash-row` class so `_focus_library_list_entry`'s
   `.library-media-row` query never grabs trash rows), Restore action with
   `library_disabled_action_label` "○" + F-018 reason when empty/loading.
   CSS: extend the `.library-media-row` selectors with the trash class in the
   source tcss, rebuild the bundle.
4. **Data**: fetch via the EXISTING seam
   `media_reading_scope_service.list_media_trash(mode="local")`; extend the
   local service's SELECT with `trash_date` (passthrough in
   `_build_local_media_list_response`) so rows can say "trashed <age>".
5. **Restore (AC#2)**: per selected item through
   `media_reading_scope_service.restore_media_item(mode="local")` — the
   already-implemented seam over `MediaDatabase.restore_from_trash` (Task 2
   contract: entering it from the Trash surface IS the explicit restore
   decision). The restore worker JOINS the shared
   `_library_media_bulk_delete_in_flight` flag + `library_media_bulk_delete`
   worker group (PR-1473 one-interlock rule: it mutates the same
   records/counts/receipt state). Completion updates
   `_local_source_records["media"]` / `_local_source_counts["media"]` in place
   and full-recomposes (rail count repaint), mirroring the Undo tail.
6. **Copy (AC#3)**: both confirm copies (bulk + single viewer) point at Trash
   ("You can undo right away, or restore later from Trash."); the receipt
   appends the durable path ("✓ deleted · N items · in Trash") per ADR-055
   Pattern A / task-14901's hand-off note.
7. **Footer/F1 + Escape**: new `escape → library_media_trash_back` binding with
   a `check_action` gate on the trash view; footer context reuses
   `LIBRARY_DETAIL_BACK_SHORTCUTS` via the one shared builder.
8. **FTS decision** made explicit (see Notes) + real-DB pins. Real-DB tests for
   restore incl. a chunked item; pilot tests for the canvas; handler tests
   mirroring the bulk-delete suite.
9. **Live-verify AC#4** (tmux, isolated HOME/config): delete → dismiss receipt
   → Trash → restore → back in list with rail/list counts right.
10. Docs (`Docs/User_Guide/library/media-and-conversations.md`) + follow-up
   task for permanent-delete/empty-trash actions (seams exist, out of AC).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Mechanism (AC#1).** The Trash view is the media canvas's third in-canvas
view (`_library_media_view = "trash"`), entered via a "Trash" toolbar
action on the media list and exited via "‹ Media"/Escape. The alternatives
were rejected for the reasons in the plan above (a `type:` cycle value
mixes STATE into a content-type cycler and collides with task-14902; a
rail row promotes media-only trash to a sibling of the Browse stores while
notes/prompts/collections have no trash surface, ADR-055 inventory). The
"Trash" button is always enabled (no count fetch on every snapshot; empty
Trash shows honest empty copy) and hidden in select mode like "Export…".

**What shipped.**
- Pure state: `LibraryMediaTrashRow`/`LibraryMediaTrashState`/
  `build_library_media_trash_state` in `Library/library_media_state.py`
  (seam-order preserved, "trashed <age>" secondary, loading/error/empty
  honesty ordering, "showing X of N" truncation line).
- Widget: `Widgets/Library/library_media_trash_canvas.py` — heading
  ("‹ Media" + "Trash (N)"), notice line, status line, `▸ `-marked rows
  (own `library-media-trash-row` class, comma-joined into the media-row
  CSS rules, so `_focus_library_list_entry`'s `.library-media-row` query
  can never grab a trash row), and a Restore action with "○" marker +
  F-018 reason tooltips (loading vs. empty).
- Seam extension: `list_media_trash` now SELECTs `trash_date` (it was
  already the sort key) and `_build_local_media_list_response` passes it
  through conditionally, same contract as `last_modified`.
- Screen wiring: compose branch + `_sync_library_canvas("media-trash")`
  in-place updater (one shared state builder, recompose discipline);
  fetch worker (`library_media_trash_load` group, read-only so it does
  NOT claim the delete interlock); Escape binding + `check_action` gate +
  footer via the one `_library_footer_shortcuts_for_current_state` seam
  (trash shares LIBRARY_DETAIL_BACK_SHORTCUTS' honest "esc back to
  list"); session-transient across app restarts and rail re-entry.

**Restore (AC#2) and the interlock.** Restore goes through
`media_reading_scope_service.restore_media_item(mode="local")` →
`MediaDatabase.restore_from_trash` — pressing Restore on the Trash
surface IS task-4026's explicit restore decision. The restore worker is
the FOURTH mutator of the shared `_local_source_records["media"]` /
`_local_source_counts["media"]` state, so it claims the SAME
`_library_media_bulk_delete_in_flight` flag and the SAME exclusive
`library_media_bulk_delete` worker group as bulk delete/Undo/single
delete (PR-1473's one-interlock rule) — never a flag of its own.

**Restore-receipt decision: NO receipt.** ADR-055's receipts accompany
DESTRUCTION; restore is recovery. Feedback is the row leaving the Trash
list, both counts moving in place, and a transient "Restored 'Title'."
notice line (no Undo affordance — a mis-restore's way back is Delete,
which is receipted). Recorded in ADR-055's Pattern A amendment.

**FTS decision (explicit): trashed items stay OUT of search results —
already true at every reachable query path, now pinned so it cannot
regress.** The task premise ("still surfaces from full-text search
today") is true at the INDEX level only — `mark_as_trash` leaves the
FTS5 row in place (which is what makes restore instant, no re-index) —
but at QUERY time every reachable path filters `is_trash = 0`:
`search_media_db` defaults `include_trash=False` (the Library keyword
seam `LocalMediaReadingService.search_media` and the RAG
`search_media_fts5` leg both route through it without overriding), and
`RAG_Search/simplified/rag_service.py`'s direct media_fts query
hard-codes `m.is_trash = 0`. Decision: keep the exclusion (no
"say trashed state in results" labelling — trashed items simply do not
appear until restored), pinned both directions by
`test_trashed_item_excluded_from_search_until_restored` (real DB:
trashed item absent from both seams, present again after
`restore_from_trash`). Known edge, documented not fixed here: with the
optional embeddings RAG enabled, vectors indexed BEFORE a deletion stay
in the vector store until re-index (`ingestion_indexing.py` itself only
indexes `is_trash = 0` rows) — a pre-existing index-freshness property
of every mutation, not a trash-specific hole; user docs therefore say
"Library search and RAG keyword retrieval".

**url-canonicalization decision (Task 2 hand-off).** The Trash-surface
restore path is `restore_from_trash` — an `is_trash`/`trash_date` flag
flip that never touches `url`, so task-4026's one-directional
url-canonicalization edge (which lives only in
`add_media_with_keywords`'s restore-by-re-import) CANNOT occur on this
path. Pinned by
`test_restore_via_real_db_chunked_item_keeps_chunks_and_url` (url and
chunks byte-identical across trash→restore, real file-backed DB).

**AC#3.** Both confirm copies now read "You can undo right away, or
restore later from Trash." and the receipt reads
"✓ deleted · N items · in Trash" (ADR-055 Pattern A: once the durable
surface exists, the copy and the receipt name it). ADR-055 amended to
record that grammar.

**Out of AC, filed.** Permanent-delete / Empty Trash (seams
`permanently_delete_media_item`/`empty_media_trash` exist, policy-gated,
unwired) → task-15130, which must follow ADR-055 Pattern B. Note: the
backlog CLI auto-assigned id 15103 which COLLIDES with origin/dev —
renamed to 15130 (leapfrog; the known backlog-ids trap, no new lesson).

**Tests.** 19 new in `Tests/UI/test_library_media_trash.py` (5 canvas
pilot, 2 toolbar/copy pilot, 7 handler/gate, 4 real-DB restore/fetch —
chunked variant included — 1 real-DB FTS pin), 12 in
`Tests/Library/test_library_media_trash_state.py`, 1 in
`Tests/Media/test_local_media_reading_service.py` (trash_date seam).
Battery: 295 passed across the media-adjacent suites (multiselect, state,
services, Media_DB task-4026 pins); CSS bundle reproduces. Ambient
failures confirmed pre-existing at base `db733c62b`:
`test_action_show_workbench_help_includes_landing_footer_keys` (stale
title assertion vs. task-4023's "— Landing" suffix, both committed at
base) and the known skills shadow-name failure.

**Live verification (AC#4).** Isolated tmux run (scratch
`TLDW_CONFIG_PATH`, users_name `sdd_lq4`, seeded 2-item media DB):
single delete via the viewer → confirm copy "…restore later from Trash."
→ receipt "✓ deleted · 1 item · in Trash" + Media (1) in list AND rail →
Dismiss → "Trash" → "Trash (1)" listing "Beta Transcript / document ·
trashed now" → Restore → "Trash (0)", "Restored 'Beta Transcript'.",
honest empty copy, "○ Restore", rail Media (2) updated in place, focus
(┃…┃) on "‹ Media" → Escape → media list with both rows, footer back to
"esc focus rail". Bulk confirm copy also verified live ("Delete 1
selected item? You can undo right away, or restore later from Trash.").
Cleanup done; live config grepped for probe inputs — zero matches.
<!-- SECTION:NOTES:END -->
