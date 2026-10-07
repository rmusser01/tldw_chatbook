---
id: TASK-34383
title: >-
  File Notes delete-safety completion: restore refusal with export fallback
status: Done
assignee: [rmusser01]
created_date: '2026-10-04'
labels:
  - notes
  - library
  - file-notes
dependencies: []
priority: high
---

## Description

Reslice of the superseded TASK-399 B-phase under the SHIPPED ADR-029 design
(one SQLite replica, disk authority; see the 2026-10-04 TASK-399 arc
reconnaissance). Delete/restore landed in minimal form (two-press confirm, snapshot+tombstone, most-recent restore). Missing: refuse restore to occupied or missing-parent paths and offer exact-export fallback instead.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Restore refuses an occupied destination path (no-replace) with a clear reason
- [x] Restore refuses a missing parent directory with a clear reason
- [x] Both refusals offer the existing exact-export as the fallback action
- [x] Tests pin both refusal shapes and the fallback path
<!-- AC:END -->

## Implementation Plan (to be added when claimed)

ADR required: no
Reason: no new storage, schema, authority, or conflict policy — this
completes the refusal UX of the SHIPPED ADR-029 delete/restore contract
("Delete commits a recovery snapshot and tombstone before unlinking") and
reuses the task-34381 exact-export read-path as the fallback action.
Extends ADR-029's landed design within its boundaries.

1. Service (`file_notes_service.py`): `restore_file`'s occupied refusal
   carries the explicit no-replace reason; a missing parent directory
   returns a typed `missing` refusal with a clear reason instead of a raw
   ENOENT "error".
2. Workspace (`library_file_notes_workspace.py`): after either refusal the
   action line names the reason AND an "Export deleted copy" affordance
   appears in the contextual actions row; pressing it exact-exports the
   tombstone's delete revision (kind='delete', session_key=None) through the
   task-34381 `export_revision_file` machinery to a user-named absent path.
   Refusal state is cleared on successful restore/export, on a fresh delete,
   and when the deleted selection moves elsewhere.
3. Tests (red first): `Tests/Notes/test_file_notes_delete_safety.py` (both
   refusal shapes with reasons; fallback export yields the exact deleted
   bytes for both shapes) and
   `Tests/UI/test_library_file_notes_delete_safety.py` (occupied and
   missing-parent refusals reported with reasons; fallback button appears
   and exports exactly).


## Implementation Notes

Completes the minimal delete/restore that landed under ADR-029 (two-press
confirm, snapshot+tombstone, most-recent restore); no new storage or policy —
ADR check recorded in the plan above (no ADR required).

**Approach.**
- `file_notes_service.py` (`restore_file`): the occupied refusal now carries
  "Destination already exists; restore never replaces a file" and a missing
  parent returns a typed `missing` refusal ("Parent directory is missing;
  restore only writes absent paths") via a `FileNotFoundError` arm before the
  generic `OSError` one, instead of a raw ENOENT "error".
- `library_file_notes_workspace.py`: `_restore_file` left the generic
  `_complete_path_action` seam so the refusal result could drive a follow-up;
  it now runs restore under `_hold_path_transition` itself, and on an
  `exists`/`missing` refusal records `_restore_refusal_path`, names the
  reason on the action line, and advertises the fallback. A new
  "Export deleted copy" button (`#file-notes-export-deleted`) appears in the
  contextual actions row while that state is live; pressing it exact-exports
  the tombstone's `delete` revision (kind='delete', session_key=None) through
  the task-34381 `export_revision_file` machinery to a validated new path
  from the path field. Refusal state clears on successful restore, successful
  fallback export, a fresh delete, and when the deleted selection moves to
  another tombstone. New shared `_action_refusal(action, result)` reports
  non-editor failures on the action line without poisoning save state
  (also adopted by the task-34381 history executor in place of its local
  closure).

**Red → green evidence** (worktree `.worktrees/wave5-build`, Python 3.12.13).
- Service red: `python -m pytest Tests/Notes/test_file_notes_delete_safety.py
  -q` → `assert 'never replaces' in None` (occupied message absent) and
  `assert 'error' == 'missing'` (missing parent fell into the generic error).
  Green after: `3 passed`.
- UI red (workspace file reverted to HEAD via `git checkout HEAD -- <path>`,
  service changes kept): `python -m pytest
  Tests/UI/test_library_file_notes_delete_safety.py
  Tests/Notes/test_file_notes_delete_safety.py -q` → `2 failed, 3 passed`
  (both UI affordance tests; the fallback button did not exist and no
  refusal reason was reported). Green after: `5 passed`.
- Combined file-notes batteries: `python -m pytest Tests/Notes/
  test_file_notes_replica.py test_file_notes_service.py
  test_file_notes_retention.py test_file_notes_delete_safety.py
  Tests/UI/test_library_file_notes_history.py
  test_library_file_notes_retention.py
  test_library_file_notes_delete_safety.py -q` → `90 passed`.
- Regression A/B: the four workspace UI batteries' FAILED list (76 names)
  is byte-identical to the pre-change baseline (pre-existing
  config-admission `RecoveryRequired` reds only).

**Design note pinned by the tests.** A tombstoned path that REAPPEARS on
disk stops being "Recently deleted" on the next scan (the replica row
returns to active — disk authority), so the occupied refusal the UI test
drives recreates the file after initialization with polling idle; the
service test pins the same shape directly.

**Files changed.** `tldw_chatbook/Notes/file_notes_service.py`,
`tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`,
`Tests/Notes/test_file_notes_delete_safety.py` (new),
`Tests/UI/test_library_file_notes_delete_safety.py` (new).

## Qodo review round (PR #3016) — 2026-10-06

Disposition of the delete-safety findings against this task's code
(restore refusals + the Export deleted copy fallback).

- **M8 — a refused-restore marker survived root changes.** Fixed.
  `_restore_refusal_path` was cleared only by a successful restore or by
  selecting a DIFFERENT relative path, so after adopting a new root the
  Export deleted copy fallback stayed visible for a same-named tombstone
  whose restore was never refused. `_commit_root_candidate.publish()` now
  clears the marker whenever a root candidate is adopted (the marker is
  scoped to the refusing root's session). UI pin (red pre-fix):
  `test_restore_refusal_marker_does_not_survive_a_root_change` — the
  fallback stayed hidden on the new root's same-named tombstone and
  reappeared only after that root's own refusal.
- **HIGH 2's fallback half.** The Export deleted copy action calls
  `export_revision_file(kind="delete", session_key=None)`, which now
  serves the newest deletion's bytes (see 34381's section); the
  "Exported the deleted bytes exactly" receipt can no longer be produced
  from an earlier deletion cycle's bytes, and the receipt appends any
  durability warning (M9).

Evidence: `Tests/UI/test_library_file_notes_delete_safety.py` (3 tests,
including the new root-change pin) green after the fixes.
