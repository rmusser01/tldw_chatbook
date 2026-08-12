---
id: TASK-15100
title: Notes delete leaves an Undo receipt per ADR-055
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 06:20'
labels:
  - library
  - ux
  - adr-055
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
- [x] #1 A successful note deletion leaves an in-place receipt in the notes list naming what was deleted, with Undo and Dismiss, matching the media receipt grammar
- [x] #2 Undo restores the note through the service seam (no raw SQL) and the list and rail Notes count update in place
- [x] #3 Undo and any concurrent note delete/create cannot race on shared list/count/receipt state (one shared in-flight interlock, PR-1473 pattern)
- [x] #4 The note delete confirm copy promises exactly what exists (Undo), replacing "This cannot be undone from Library."
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: `backlog/decisions/055-library-destructive-action-reversibility-rule.md`
Reason: ADR-055 already defines the Notes recovery receipt, restore boundary, and shared mutation interlock; this task directly implements that accepted contract without changing an architectural boundary.

1. Characterize the current version-checked Notes delete path and add a matching restore operation through `ChaChaNotes_DB`, `Notes_Library`, and `NotesScopeService`, including Sync v2 enqueue behavior.
2. Add focused store/service tests for successful restore, stale-version conflicts, policy enforcement, and Sync v2 restoration events.
3. Add mounted Library tests for the receipt grammar, confirm promise, Undo/Dismiss behavior, list/count restoration, supersession, and shared create/delete/Undo admission.
4. Render the persistent receipt in the Notes list and connect delete, Undo, Dismiss, focus recovery, and count/list refresh through one shared Notes mutation interlock.
5. Update the Notes Library guide, run focused tests and static checks, self-review the diff, and complete the task record with verified evidence.

## Implementation Notes

- Implemented ADR-055 Pattern A for Database Notes: confirmed deletion now patches the visible list/count immediately and leaves a named `✓ deleted · <title>` receipt with Undo and Dismiss. The confirmation copy now promises the Notes-list Undo that actually exists.
- Added a version-checked restore path through `CharactersRAGDB.restore_note` → `NotesInteropService.restore_note` → `NotesScopeService.restore_note`; the screen never uses SQL. Restore returns the fresh active row and preserves keyword metadata for Sync v2. Legacy Notes FTS update triggers are detected and repaired once during DB initialization, keeping schema DDL out of the user-triggered Undo transaction.
- Serialized create, visible delete, untouched-blank GC/discard, and receipt Undo through one `library_note_mutation` worker group and `_library_notes_mutation_in_flight` admission flag. While Undo is pending, row entry, Create, Dismiss, and another delete are refused; list/count/receipt patches finish before the flag is released.
- Qodo review hardening made soft delete strictly optimistic: a repeated or concurrent delete of an existing tombstone now raises `ConflictError` rather than reporting a second success. Therefore the UI only decrements the Notes count and creates an Undo receipt when its call actually performs the active-row-to-tombstone transition, and the derived tombstone version is exact. Public restore/receipt callables also now document their arguments, return values, and raised errors in Google style.
- Updated `Docs/User_Guide/library/notes.md` and recorded the Windows `pytest-asyncio`/network-guard setup failure in `backlog/docs/lessons-testing-evidence.md`. No new ADR was required; `backlog/decisions/055-library-destructive-action-reversibility-rule.md` is the governing decision.
- Verification: 47 Notes scope-service/real-SQLite tests passed after review hardening, including repeated-delete conflict, restore-time zero-DDL, and one-time legacy-trigger-repair regressions; 6 mounted delete/receipt/Undo/interlock tests passed, including both success and failure branches. Earlier task verification also passed 19 mounted Notes create/delete/Undo/discard/blank-GC regressions and 63 Notes state, multiselect, and CSS integrity tests. `compileall`, focused Ruff checks (excluding the file's pre-existing E721/F401 debt), and `git diff --check` passed. On Windows, current `dev`'s task-15111 socket guard blocks Python's own Proactor self-pipe before async tests start, so focused local-only suites ran with that guard's family set disabled only inside the pytest process; all selected tests use SQLite or injected fakes and no external clients.
