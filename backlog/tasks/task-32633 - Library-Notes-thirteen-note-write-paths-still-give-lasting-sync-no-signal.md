---
id: TASK-32633
title: >-
  Library Notes: thirteen note-write paths still give lasting sync no signal
status: To Do
assignee: []
created_date: '2026-09-15 10:15'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider off task-32604 (the critique-#4 P0). 32604 closed the reachable hole for
an editor save of an already-bound note, and made a root whose runtime has
stopped wear "⚠ Sync stopped" instead of "✓ Up to date". It did NOT cover the
other note-write paths: the enumeration in 32604's notes counts **14 write
sites, 1 covered, 13 not — 5 updates and 8 creates**.

The two halves need different mechanisms, which is why 32604 left them here:

- The **5 update paths** write to an already-bound note, so they are one
  `note_changed(note_id)` call away from correct — the seam exists and is
  already proven by 32604's covered path.
- The **8 create paths** mint a note that no binding knows about yet, so
  `note_changed` cannot resolve it to a root. They need a folder-membership
  predicate, not a binding predicate.

Until then those paths leave the file on disk stale while the row reads
"✓ Up to date" on a live runtime, recoverable only by **Check changes** or a
disk-side change. This rider also carries Minor 7 from 32604's re-review
(the per-watchable-root read is a shape worth revisiting, acknowledged not
changed) and the deletion-group count nit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A save whose signal lasting sync refuses is visible to the user at the editor, not silent: the note session surfaces it rather than leaving the reassuring row to speak for the write. (Needs a new field through `DatabaseNotePortSaveReply` → `NoteSaveOutcome` → the session snapshot — the note-session state machine, which is why 32604 did not do it.)
- [ ] #2 Each of the 5 update paths signals lasting sync on success, verified per path against a live runtime with a real file on disk.
- [ ] #3 The 8 create paths are covered by a folder-membership predicate (or an explicit, user-visible decision that they are not), with the mechanism named and pinned.
- [ ] #4 The enumeration in the task notes and `Docs/User_Guide/library/notes.md` is re-derived at landing time, not copied — the count moved twice already (12+1 → 14/1/13).
- [ ] #5 Minor 7 adjudicated: either the per-watchable-root read is reshaped or the reason it stays is recorded.
<!-- AC:END -->
