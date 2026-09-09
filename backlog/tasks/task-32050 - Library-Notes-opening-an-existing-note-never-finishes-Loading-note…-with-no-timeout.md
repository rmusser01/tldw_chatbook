---
id: TASK-32050
title: >-
  Library Notes: opening an existing note never finishes ('Loading note…' with
  no timeout)
status: Done
assignee: []
created_date: '2026-09-08 18:22'
updated_date: '2026-09-08 19:05'
labels:
  - library
  - notes
  - ux
  - critique-8
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Any note opened from the Notes list stays on 'Loading note… · Next: Wait for loading to finish.' indefinitely, with no failure state, Retry or Cancel; after the first hang every later open hangs too, and only notes created in the same session are editable. Reproduced five times across both assessors and two diagnostic sessions (a 60-byte note opened first also hangs). A thread dump during the hang shows the event loop idle and no worker thread running the load, so the load outcome is dropped or the coroutine is cancelled before it paints (_refresh_library_note_detail generation guards / open_session request token / _run_library_service_call's asyncio.run inside to_thread are the suspects). The notes guide was verified working on 2026-09-06 (fix/library-uat-31796-31797), bounding the regression window. This breaks the core write-reopen-summarise loop. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening any existing note from the Notes list renders the editor with title and body on the seeded profile, including the 35 KB note
- [x] #2 A note load that exceeds a deadline (about 3 s) shows the existing failed state with Retry instead of a permanent 'Loading note…'
- [x] #3 After a failed or slow load, opening another note still works in the same session
- [x] #4 The root cause is identified in the task notes and a regression test opens a stored note through the real note session port (not a fake service)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on the seeded profile with temporary trace points through _refresh_library_note_detail / open_session / the port / the service worker; read the trace to find the last line reached.
2. Write a failing regression test that opens a stored note through the real DatabaseNoteSessionPort and a real CharactersRAGDB (no fake scope service) and asserts the editor renders the stored body.
3. Run it RED.
4. Fix the root cause at the seam that drops the completed outcome; remove the trace points.
5. Add LIBRARY_NOTE_LOAD_DEADLINE_SECONDS = 3.0 and wrap the load in asyncio.wait_for so a stuck load reaches the existing failed state with Retry; test the timeout copy and that a later note still opens.
6. Run both new tests plus the coordinator/session test files.
7. Live-verify on the seeded profile (35 KB note and a short note, Escape back, re-open).
8. Docs stamp in Docs/User_Guide/library/notes.md, tick ACs, notes, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (traced live on the seeded profile with temporary trace points, then reproduced in the harness): the note-detail load always SUCCEEDED and was then thrown away. `_refresh_library_note_detail` was fenced by the Notes *tree* tokens (`_library_notes_navigation_generation` plus the topology/lifecycle epochs) as well as by its own identity. `_begin_library_note_load` starts two workers: the detail load (capturing generation N) and the folder-tree locator, which sets `_library_notes_navigation_status = 'Locating note…'`. `on_descendant_focus` bumps the navigation generation to N+1 whenever a user-intent focus lands while that status is set -- which is exactly what the row click that started the load does, ~20 ms in. The load returned LOADED ~110 ms later, the second guard saw N != N+1, returned SUPERSEDED, and nothing ever left `_library_note_load_state == 'loading'`. Live trace: 'supersede -> 4 ... on_descendant_focus:10852' between 'before open_session' and 'guard2 SUPERSEDED nav=3/4'.

Fix: the detail load no longer takes the tree tokens at all. It is already superseded three ways that key on its own identity -- the exclusive `library_note_detail` worker group, the coordinator's session request token (STALE outcome), and `_selected_note_id`/view/source. The two other callers (the mount-time deep link and Retry) already passed no tokens, so this makes all three consistent. Deleting the parameters was smaller than special-casing the focus seam and fixes the retry path too.

Deadline: `LIBRARY_NOTE_LOAD_DEADLINE_SECONDS = 3.0` and `LIBRARY_NOTE_LOAD_TIMEOUT_COPY` in `screen_constants.py` (imported into `library_screen.py`); `open_session` runs under `asyncio.wait_for`, and a timeout projects the existing failed state -- 'Unable to load note — timed out after 3 s. Press Retry.' -- through `_project_library_note_entry_result`, so the canvas's existing Retry button appears.

Tests: `Tests/UI/test_library_crit8_notes_loader.py` (new, 2 tests) runs the production wiring -- real `CharactersRAGDB`, real `NotesInteropService`/`NotesScopeService`, therefore the real `_LibraryDatabaseNoteSessionPort` -- and clicks the real row so the focus event that caused the bug actually fires. RED before the fix: 'note editor never rendered the stored body; load state is loading'. The second test stalls the port's `load_note` and asserts the timeout copy, the Retry button, and that a later note still opens.

Live-verified on the seeded profile at 235x52: Reading list opens with title and body, Escape returns, the 35 KB 'Very long note' opens, Escape returns, Reading list re-opens. Captures under scratchpad/crit8/wave/notes-loader/caps/.

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/screen_constants.py, Tests/UI/test_library_crit8_notes_loader.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
