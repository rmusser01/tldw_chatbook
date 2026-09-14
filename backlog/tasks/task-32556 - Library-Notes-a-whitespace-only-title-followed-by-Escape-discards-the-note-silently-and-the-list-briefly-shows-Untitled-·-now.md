---
id: TASK-32556
title: >-
  Library Notes: a whitespace-only title followed by Escape discards the note
  silently, and the list briefly shows "Untitled · now"
status: Done
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 19:25'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, persona Riley. Residual of task-32133 (blocked-save Escape veto).

**What happened.** New note, whitespace-only title, Escape: the editor leaves to the list with no "Can't leave yet" toast and no receipt; the DB row is `Untitled`, content 0, `deleted=1` (discarded, as the guide documents for an untouched blank note); the list momentarily still showed "Untitled · now" (B 55 line 39 + DB; A 66). Captures: A 66; B 55.

**Cause.** PROVEN by DB read; the discard is documented behaviour, the silence is not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Leaving a blank note whose only content is a whitespace title shows the documented "Can't leave yet…" prompt, or a one-line "Empty note discarded" receipt
- [x] #2 The list never paints a row for a note that is about to be discarded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce (done: whitespace title + Escape discards silently; DB row deleted=1 while the list keeps painting 'Untitled · now' - the ghost persists, it is not momentary).
2. RED pins: the discard receipt, and no projection row for the discarded note.
3. Fix: notify 'Empty note discarded' when the GC branch fires on a non-empty raw title; reconcile the folder tree after the GC delete the way the visible Delete does, so the row leaves the projection.
4. GREEN + live.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live at dev 2f97a42c9a (235x52), and the list half is worse than
filed: Ctrl+N, three spaces in Title, Escape — no toast, and the list kept
painting "Untitled · now" four seconds later for a row `sqlite3` already
showed as `deleted=1`. Not "briefly": it persisted.

**AC#1.** `_flush_library_note_save`'s GC branch now distinguishes a note the
user never touched (discarded in silence, as the guide documents) from one
whose title was TYPED and happens to be blank. The second notifies "Empty
note discarded". Keyed on `raw_title` being non-empty but blank after
stripping, which is exactly the case the task describes.

**AC#2 — root cause.** The list is projected from the paged folder tree, not
from the flat cached source records. `_create_library_note` reconciles its
new note INTO that tree ("note_create"); `_gc_pending_blank_note` removed
only the flat record, so the row survived in the projection until the next
full tree reload. The GC now commits the matching "note_delete" through the
same reconciler the visible Delete uses — the one seam, not a second
mechanism — and because the GC is awaited before the editor exit flips the
view to "list", the list is re-projected only after the tree is correct.

**Tests.** `::test_a_whitespace_only_title_then_escape_shows_the_discard_
receipt` asserts the notify text through the real Escape route;
`::test_the_list_never_paints_a_row_for_a_note_being_discarded` asserts, over
six refreshes, that the blank note's id appears in neither the production
tree projection nor any mounted row. Both RED on detached origin/dev, GREEN
here. Live at 235x52 and 100x30
(`editor-10-32556-empty-note-discarded-235x52`,
`editor-10-32556-empty-note-discarded-100x30`); the profile database shows
the row `deleted=1` with no row painted and the rail count correct.

Modified: `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, the pin file,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
