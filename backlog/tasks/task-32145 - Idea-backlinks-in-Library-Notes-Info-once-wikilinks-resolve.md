---
id: TASK-32145
title: 'Idea: backlinks in Library Notes Info once wikilinks resolve'
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 17:33'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - idea
  - obsidian
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improvement for the researcher persona, downstream of task-32129: 'Linked from (N)' in Info → Properties is nearly free once `[[wikilinks]]` become note links, and is the feature that makes an imported vault feel adopted rather than copied. Size M. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Depends on task-32129 landing
- [x] #2 Design agreed with the user before implementation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm link form: note_import_plan_models writes [label](note://<note_id>).
2. DB: CharactersRAGDB.get_notes_linking_to(note_id, limit) -- parameterised LIKE '%(note://<id>)%' with ESCAPE, deleted = 0, id != self, ORDER BY title, LIMIT ?.
3. Notes_Library passthrough + NotesScopeService.list_note_backlinks (asyncio.to_thread, local scope only).
4. LibraryNotesState.backlinks field; controller loads it in a worker off _begin_library_note_load, then _apply_library_note_presentation_state.
5. LibraryNotePresentationState.backlinks -> Info Properties 'Linked from (N)' + one row Button per backlink (composed, and reconciled in apply_session_state so it paints without a recompose).
6. Screen handler .library-note-backlink -> flush save + _begin_library_note_load.
7. Tests: Tests/Notes/test_note_backlink_query.py (real DB) + Tests/UI/test_library_notes_riders_backlinks.py; guide stamp; commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Info → Properties gained 'Linked from (N)': the notes whose bodies carry this note's link, each entry opening that note.

AC#1 (depends on task-32129) is satisfied — the Obsidian importer already writes the link form this reads. AC#2 (design agreed) is ticked on the controller's recorded ruling on the user's delegation of this wave.

Link form: `[label](note://<note_id>)` (note_import_plan_models.rewrite_wikilinks). The query matches the closing parenthesis too, so `note://abc` cannot also match a link to `note://abcdef`.

Query (new `CharactersRAGDB.get_notes_linking_to`): parameterised LIKE with the id's own LIKE wildcards escaped (`ESCAPE '\\'`), `deleted = 0`, the target itself excluded, ordered by title, LIMIT bound. FTS5 was the wrong index: its tokenizer splits `note://<uuid>` into `note` plus hex runs, so a MATCH would answer a looser question. Reached through a `NotesInteropService` passthrough and `NotesScopeService.list_note_backlinks` (asyncio.to_thread, local scope only — every other scope answers empty rather than raising into a panel).

Load: its own worker off `_begin_library_note_load`, not part of the detail load — the detail load owns how fast the editor appears. It asks for cap+1 rows so an over-cap result reads '50+' instead of claiming an exact 50. Stale results are dropped by the same selected-note/view guard the detail load uses, and the state field is reset when a note opens.

Paint: BOTH compose and `apply_session_state`, sharing one `_backlink_buttons` builder. The reconcile is what makes them appear at all — the rows land after the editor is composed, a recompose is deferred while the reader owns a field (task-32062), and Edit→Info is a display flip on the same composition rather than a rebuild. Pinned by test_late_arriving_backlinks_paint_without_a_recompose (verified load-bearing: disabling the reconcile fails exactly that test).

Activation: `.library-note-backlink` handler on the controller (screen delegates in 3 lines — library_screen.py is already ~1.3k lines over its ratchet budget), same flush-then-open contract as a list row minus the list-only concerns.

Files: DB/ChaChaNotes_DB.py, Notes/Notes_Library.py, Notes/notes_scope_service.py, UI/Library_Modules/library_notes_state.py (+1 field, wiring pin 100→101), UI/Library_Modules/library_notes_controller.py, UI/Screens/library_screen.py, Widgets/Library/library_notes_canvas.py, Docs/User_Guide/library/notes.md, Tests/Notes/test_note_backlink_query.py (new), Tests/UI/test_library_notes_riders_backlinks.py (new), Tests/UI/test_destination_shells.py (fake seam mirroring the real signature).

Live-verified on the fresh profile after importing the review vault (59 notes): 'Zettelkasten — overview' read 'Linked from (2)' listing both linking notes; activating one opened it; 'scratch' read 'Linked from (0) — no notes link here yet'.

No CSS touched (the rows reuse library-canvas-action), so no build_css run.

Review round (PR #2552, Qodo): the header no longer claims a count the query
has not answered. `backlinks_status` ("loading"/"ready"/"failed", wiring pin
101→102) makes a pending lookup read "Linked from — checking…" and a failed
one (raising query, or no service to ask) "Linked from — couldn't check",
so only a completed query can say "no notes link here yet". Screen handler
got its docstring. Rebutted: the shared-constant ask (the one production
caller passes `LIBRARY_NOTE_BACKLINK_DISPLAY_CAP + 1` explicitly; the DB and
service `limit=50` defaults are a bound for direct callers, and a
dependency-neutral module for one integer is not worth its own file); the
`transaction()`/cursor-context asks (69 sibling reads in ChaChaNotes_DB.py
use the same `execute_query` + `fetchall`, and `execute_query` already joins
an enclosing transaction when there is one); the in-memory-DB test ask
(CharactersRAGDB keeps thread-local connections, so `:memory:` gives each
thread its own empty database — the query crosses a thread through
`asyncio.to_thread` and dies with "no such table: notes", verified). The
full-scan performance finding is real at vault scale and deferred as
task-32186 (it needs a persisted link relation and a migration).
<!-- SECTION:NOTES:END -->
