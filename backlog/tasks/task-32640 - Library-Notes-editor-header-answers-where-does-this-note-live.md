---
id: TASK-32640
title: >-
  Library Notes: editor header answers "where does this note live?"
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
Critique #4 idea 1, ACCEPTED in task-32627. One row under the note title
reading which of the three worlds this note belongs to and, for a synced
note, its file path and when it was last written to disk.

This is the cheapest answer to a theme every critique has raised: the three
worlds (database-only, managed folder, edit-in-place) are not answerable from
inside a note, so a user cannot tell whether what they are typing will reach a
file. It is also the place a stale or refused write becomes visible without a
trip to Manage — related to task-32633, which owns the refusal signal itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An open note shows, without tabbing or opening Info, which world it lives in, in the user's words rather than an internal name. SHIPPED SCOPE: on a terminal 80 columns or wider. Below that the row is given back to the body on the same rule the chrome strip already uses — see the trade-off in the Implementation Notes.
- [x] #2 A synced note also shows its file path and when the file was last written; a database-only note shows neither and does not pretend to.
- [x] #3 The row costs at most one line and degrades at 100x30 — it truncates the path, never the world.
- [x] #4 The line is derived from live state, not from what the note looked like when it was opened.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find the live answer: which binding claims this note, and which root path it sits under.
2. Add the note-side read the sync store never had, and the runtime join that turns it into a path.
3. Resolve it in its own worker per note open (and after a save), like the backlink lookup beside it.
4. One Static under the title, one pure width-aware line builder, re-stated on resize.
5. RED-first pins at all three critique sizes; guide + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The missing read was on the note side.** Every binding query in
`NotesDeviceStateStore` starts from a ROOT (`active_binding_note_ids`,
`has_binding_for_note_or_path`, `list_bindings`); nothing could answer "which
file is this note in" without walking every root and hydrating rows it would
discard. `active_binding_path_for_note` is one indexed-free `LIMIT 1` read on
`note_id AND state = 'active'`, and `NotesSyncRuntimeOwner.note_file_location`
joins it to `_root_paths` — the map `_start_once` fills from the same records
the watcher leases. **No index was added**: this store has no `sqlite_stat1`
either (the migrations README's rule), and one scan per note opened does not
justify the census a `CREATE INDEX` drags behind it.

Every LOADED root counts, not only the watchable ones. A paused or blocked
root still holds the file the note lives in; whether anything is watching it is
the root row's job (task-32604 made that row honest), not the header's.

**Live, per AC#4.** The worker runs on note open and again when a save
settles, and both facts are read fresh at that moment: the binding from the
runtime, the write time from `os.stat` on the file itself. The mtime is
deliberately the file's own rather than a record of our writes — a vault
edited in Obsidian and a note saved here are the same question to the reader,
and only the filesystem answers both. Nothing is stored on the note, so a note
that is bound, retargeted or disconnected while open cannot leave a stale
sentence on screen.

**Known ceiling, corrected in review round 1 — the first version of this
paragraph claimed the opposite of the code comment.** The post-save re-read
RACES the write. A save only `schedule_hint`s the root (task-32604) and the
file is written by a later background pass, so the `os.stat` usually happens
BEFORE the bytes land, and nothing re-runs afterwards: right after your own
save the row normally still names the PREVIOUS write time, until the note is
reopened or saved again. That reading is accurate — the file really has not
been written yet, which is the honest half of what this row exists to show —
but it is not "fresh after every save", and neither these notes nor the guide
may say that it is. The same staleness applies to an EXTERNAL write while the
note sits open. Re-running the worker off the sync pass itself would close
both; that is task-32633's ground, not this task's.

**The body-row cost, re-measured in review round 1.** This row costs the
editor's body one row at every size it is shown. Together with task-32642's
Keywords row the wave takes six of 29 body rows at 235x52 (23 left), two of 15
at 100x30, and one of 6 at 60x20 where this row is gated off; the full
dev-vs-branch table is on task-32642.

**The narrow-terminal trade.** Measured at 60x20 on the real screen
(`#library-note-body.region.height`): dev 6 rows, and 4 with this row plus
task-32642's Keywords row shown — 5 with it gated off, which is what ships. A note editor with two rows of note is a worse answer to "can I
work here" than an unanswered "where does this live", so below 80 columns the
row is given back to the body — the same threshold and the same reasoning as
the chrome strip under it (task-32143). Nothing else states the world at that
width; recorded rather than hidden.

**Copy.** "In the Library database only — no file on disk" reuses the Notes
list header's own noun for that world; "In a synced folder" is the chooser's
(Keep a folder synced). Neither is an internal name, and the row states the
world first and never elides it.

**Review round 1 added two paint assertions** that the location row is one
painted row and that the property/keywords geometry is what the sheet claims —
the CSS half of this wave was asserted only through `renderable`, the string
the widget is handed, which cannot see the pane at all.

**Evidence.** Store and runtime against a real `NotesDeviceStateStore` with a
real root and binding (`Tests/Notes/test_notes_sync_note_location.py`),
including the negative control that a CANDIDATE binding is not a location.
Header on the real mounted screen at 235x52. REDs recorded on a reverted COPY:
`NoMatches: No nodes match '#library-note-location'` with the row removed, and
`In a synced folder · /private/var/…/…Sam.md` — no write time — with the
`stat` removed.

**Files.** `Notes/notes_device_state_store.py`, `Notes/notes_sync_runtime.py`,
`UI/Library_Modules/library_notes_controller.py`,
`Widgets/Library/library_notes_canvas.py`,
`css/components/_agentic_terminal.tcss` (+ generated sheet),
`Tests/Notes/test_notes_sync_note_location.py` (new),
`Tests/UI/test_library_notes_w5_ideas.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
