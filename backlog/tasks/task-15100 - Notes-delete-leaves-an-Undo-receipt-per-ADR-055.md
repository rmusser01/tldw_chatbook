---
id: TASK-15100
title: Notes delete leaves an Undo receipt per ADR-055
status: To Do
assignee: []
created_date: '2026-08-11 06:20'
labels:
  - library
  - ux
  - adr-054
dependencies: []
priority: medium
---

## Description

ADR-055 (task-14901) sets one reversibility rule across Library destructive
actions: soft-deleting persisted user data leaves a receipt at the point of
action offering Undo, with the durable recovery story named. Notes deletion is
a soft delete (`notes_scope_service.delete_note` → `soft_delete_note` in
ChaChaNotes, version-checked), so the deleted row is recoverable at the store
level — but the UI today goes confirm-then-silence: after
`_delete_library_note` succeeds, the editor resets to the list with no receipt
and no way back. The confirm copy is honest ("This cannot be undone from
Library."), which ADR-055 accepts only as the interim state; the owed pattern
is the media one (task-4022): a "✓ deleted" receipt with Undo/Dismiss in the
notes list, restoring the exact row (an un-delete through the service seam,
never raw SQL — a restore seam may need to be added to
`notes_scope_service`/`Notes_Library` first). Mind the session coordinator: the
delete runs under a `DestructiveAdmission`, so Undo must not race a new
session's edits (the PR-1473 one-flag lesson applies: every mutator of the
shared list/count state joins one interlock). Update the confirm copy to
promise the Undo once it exists.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A successful note deletion leaves an in-place receipt in the notes list naming what was deleted, with Undo and Dismiss, matching the media receipt grammar
- [ ] #2 Undo restores the note through the service seam (no raw SQL) and the list and rail Notes count update in place
- [ ] #3 Undo and any concurrent note delete/create cannot race on shared list/count/receipt state (one shared in-flight interlock, PR-1473 pattern)
- [ ] #4 The note delete confirm copy promises exactly what exists (Undo), replacing "This cannot be undone from Library."
<!-- AC:END -->
