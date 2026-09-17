---
id: TASK-32641
title: >-
  Library Notes: say a vault was recognised before the import review, not inside it
status: Done
assignee: []
created_date: '2026-09-15 10:35'
labels:
  - library
  - notes
  - critique-4
  - idea
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 3, ACCEPTED in task-32627. On folder selection, detect an
Obsidian vault and say so on the confirmation line — with the counts and what
will be skipped — instead of leaving the user to discover it inside the review.

The detection already exists; only the surfacing is missing. This turns the
Obsidian toggle from a checkbox the user has to interpret into a recognised
handshake, and it is the moment at which the duplicate-vault class of bug
(task-32605, task-32637) becomes visible to the person who can still change
their mind.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Selecting a folder that is an Obsidian vault says so before the review, naming what was detected.
- [x] #2 The line carries the counts and names what will be skipped (`.obsidian/`, `.trash/`, `Templates/`, empty files) in user terms. SHIPPED SCOPE: the COUNT is of notes the scan would read, and of files already imported; the skips are named as the rule the import follows — the vault folders when the scan found them, and empty files always, because they always are. A count of empty files would need the parse the review runs, which is the thing this line exists to precede.
- [x] #3 A folder that is NOT a vault says nothing extra — no empty "0 detected" row.
- [x] #4 If any of those files were already imported or already bound, the line says so here, where the user can still choose differently.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Call the existing vault predicate, not a new one; scan with the same bounded discovery the review's check runs.
2. Count what the scan found; read "already imported" from Import once's own receipt ledger.
3. Ask the sync side whether a root already covers this folder — the screen owns both, so it asks and hands the answer over.
4. One Static under the confirmation, composed only when there is something to say.
5. RED-first pins over a real vault on disk; guide + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**One predicate, one scanner.** The recognition calls
`folder_is_obsidian_vault` -- the shared one -- and then runs
`discover_import_sources` with the SAME bounds the review's check uses, so the
counts on the confirmation line and the groups in the review cannot become two
opinions. The vault-owned folders are named from the skip reason codes
discovery already records (`obsidian_config` / `obsidian_trash` /
`obsidian_template`), through one map, so a fourth skipped folder there cannot
go unnamed here. If the marker says vault and the scanner disagrees, the line
says nothing rather than claim a vault the import will not read as one.

**Where the two halves of AC#4 come from.** "Already imported" is Import once's
own receipt ledger (`prior_imported_notes_read_only`), read over the sources
the scan just found -- the same read task-32605 gave the sync planner.
"Already bound" is the sync side, which the import controller cannot see and
should not learn to: `LibraryScreen` owns both, so it answers
`_library_folder_is_sync_root` and hands the boolean over.
`NotesSyncRuntimeOwner.folder_is_sync_root` is deliberately "covered by", not
"equal to" -- importing a sub-folder of a synced vault duplicates it just as
thoroughly.

**Cost.** One bounded read-only scan per folder pick, on a thread, published
when it lands; the selection paints immediately either way. A folder that is
not a vault costs one `is_dir`. The scan is run twice in a full Import once
(here and at Check) -- rejected the alternative of caching the discovery for
the check to reuse, because `check()` re-runs it whenever the Obsidian toggle
flips and a cache keyed on that is more moving parts than a second read-only
walk.

**Scope note.** Everything the line says is stated before the review; nothing
about the review itself changed, and the Obsidian toggle stays where it is.
The recognition is dropped whenever the selection changes or is cleared, so it
always describes the folder that was scanned.

**Evidence.** Production discovery and a real (empty) receipt ledger over a
vault on disk shaped like the critique's own -- `.obsidian/`, `.trash/`,
`Templates/` and two real notes -- plus the same walk through the real screen's
`_accept_library_note_import_path`, which is the whole of what the picker's
callback does once the dialog closes. REDs on a reverted COPY:
`AttributeError: 'LibraryNoteImportController' object has no attribute
'recognise_selected_folder'` (four tests) and `NoMatches: No nodes match
'#note-import-vault-recognition'` (three sizes). The non-vault case is the
vacuity guard: the two folders differ only by the `.obsidian/` marker.

**Not verified live in the TUI.** The picker dialog itself was not driven; the
experiment that would settle it is a scratch-profile run of Library ▸ Notes ▸
Add from files ▸ Import once against a real vault, reading the row under the
confirmation. Everything below that dialog is pinned on the real screen.

**Files.** `Library/library_note_import_state.py`,
`UI/Library_Modules/library_note_import_controller.py`,
`UI/Screens/library_screen.py`, `Notes/notes_sync_runtime.py`,
`Widgets/Library/library_note_import_canvas.py`,
`Tests/UI/test_library_notes_w5_ideas.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
