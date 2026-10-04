---
id: TASK-34381
title: >-
  File Notes revisions read-path: history list, verify, exact export, restore
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
reconnaissance). The `revisions` table in `file_notes_replica.py` is written (one coalesced checkpoint per session via `INSERT OR IGNORE`) but has NO reader anywhere: no history list, verify, export, or restore API or UI. This task adds the bounded read-path.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] A replica read API lists revisions for a file (time, size, hash), bounded (most recent N)
- [x] A revision can be verified against its stored hash and exact-exported via the existing exact-export machinery
- [x] A revision can be restored to an absent path with no-replace semantics (O_EXCL), refusing occupied paths
- [x] A bounded History affordance exists in the Folder-files workspace for protected files (list + verify + export + restore)
- [x] Targeted tests cover list/verify/export/restore incl. occupied-path refusal and hash-mismatch refusal
<!-- AC:END -->

## Implementation Plan (to be added when claimed)

ADR required: no
Reason: extends the SHIPPED ADR-029 replica (one owner-secured `file_notes.sqlite`,
disk authority) strictly within its boundaries — a read-path for the already-written
`revisions` table plus no-replace restore/export. No new storage authority, schema
migration, or conflict-policy change (references `backlog/decisions/029-file-notes-disk-authority.md`).

1. Replica read API (`file_notes_replica.py`): `ReplicaRevisionInfo`/`ReplicaRevisionBytes`
   types; `list_revisions` (time/size/hash, most-recent N, bounded), `get_revision`
   (kind+session_key selector), `verify_revision` (sha256 vs stored hash, `None` when absent).
2. Service read-path (`file_notes_service.py`): `list_revision_history` (bounded,
   warning-carrying), `verify_revision` (typed `OperationResult`), `export_revision_file`
   (exact-export machinery: same-dir temp, digest-vs-stored-hash verify, `os.link`
   no-replace, parent fsync; refuses occupied + hash mismatch), `restore_revision`
   (digest verify, `O_EXCL` no-replace create, refuses occupied + hash mismatch).
3. Workspace affordance (`library_file_notes_workspace.py`): protected-files History
   button in the maintenance row; `FileNotesHistoryDialog` modal (bounded list,
   destination input, Verify/Export/Restore) returning a request the workspace
   executes under `_hold_path_transition`.
4. Tests (red first): `Tests/Notes/test_file_notes_replica.py` (list/get/verify),
   `Tests/Notes/test_file_notes_service.py` (history/verify/export/restore incl.
   occupied + hash-mismatch refusals), `Tests/UI/test_library_file_notes_history.py`
   (dialog affordance: list + verify + export + restore).


## Implementation Notes

Reslice of superseded TASK-399 B-phase under shipped ADR-029; extends the
landed replica within its boundaries (read-path only, no schema/authority
change; ADR check recorded in the plan above).

**Approach.**
- `file_notes_replica.py`: new `ReplicaRevisionInfo`/`ReplicaRevisionBytes`
  NamedTuples plus three readers on the existing `revisions` table —
  `list_revisions` (most-recent-first, `ORDER BY created_at DESC, kind,
  session_key`, `LIMIT`-bounded, size from stored bytes),
  `get_revision` (kind + NULL-safe session_key selector via `IS ?`),
  `verify_revision` (sha256 of stored bytes vs recorded digest; `None` when
  absent).
- `file_notes_service.py`: `list_revision_history` (bounded, warning-carrying
  `RevisionHistoryResult`), `verify_revision` (typed `OperationResult`:
  ok/missing/conflict/replica-error), and a shared `_load_revision` guard
  (path safety, supported extension, size limit, stored-hash verify) feeding
  `export_revision_file` — the existing exact-export machinery shape
  (same-dir temp, digest-of-written-bytes vs stored hash, `os.link`
  no-replace, parent fsync) — and `restore_revision` (`O_WRONLY|O_CREAT|
  O_EXCL|O_NOFOLLOW`, occupied → `exists` "never replaces", missing parent →
  `missing`, then `_finish_published_file("restored", …)` so the replica and
  session changes track the new path).
- `library_file_notes_workspace.py`: `History` button in the maintenance row,
  visible only for protected documents; opens `FileNotesHistoryDialog`
  (bounded `OptionList` newest-first, destination `Input` prefilled with
  `<stem>-recovered<suffix>`, Verify/Export/Restore/Close, per-dialog status
  line). The dialog only presents and dismisses with a
  `_HistoryActionRequest`; the workspace executes it inside
  `_hold_path_transition` via a worker, reporting on the action line —
  deliberately NOT via `_operation_error`, whose conflict/missing handling
  would wrongly mark the open document conflicted.

**Red → green evidence** (worktree `.worktrees/wave5-build`, uv venv
Python 3.12.13).
- Red (APIs absent): `python -m pytest Tests/Notes/test_file_notes_replica.py
  -x -q -k revision` → `1 failed` (AttributeError: list_revisions);
  `python -m pytest Tests/Notes/test_file_notes_service.py -q -k
  "revision_history or verify_revision or export_revision or
  restore_revision"` → `4 failed` (AttributeError).
- Green: `python -m pytest Tests/Notes/test_file_notes_replica.py
  Tests/Notes/test_file_notes_service.py -q` → `74 passed`.
- New UI battery `Tests/UI/test_library_file_notes_history.py`
  (list/verify/export + occupied-refusal restore):
  `python -m pytest Tests/UI/test_library_file_notes_history.py -q` →
  `2 passed`.
- Regression A/B (pre-existing reds are the dev-tip config-admission
  `RecoveryRequired` class — every `_production_workspace_context` test is
  red at HEAD in this environment): ran
  `python -m pytest Tests/UI/test_library_file_notes_workspace.py
  Tests/UI/test_library_notes_w4_file_notes.py Tests/Notes/test_file_notes_service.py
  Tests/Notes/test_file_notes_replica.py -q` with and without this change
  (`git diff > patch; git checkout HEAD -- <paths>; …; git apply patch`);
  FAILED lists byte-identical (58 names, all pre-existing). Same for
  `test_library_notes_riders_r_file_notes.py +
  test_library_notes_wave_file_notes.py`: 18 names, byte-identical.

**Deviations.** The new UI tests mount via `_WorkspaceHarness` rather than
`_production_workspace_context` because the production-context helper is
uniformly red at HEAD here (pre-existing `RecoveryRequired`), including for
every existing w4 test — not a change this task introduced.

**Files changed.** `tldw_chatbook/Notes/file_notes_replica.py`,
`tldw_chatbook/Notes/file_notes_service.py`,
`tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`,
`Tests/Notes/test_file_notes_replica.py`,
`Tests/Notes/test_file_notes_service.py`,
`Tests/UI/test_library_file_notes_history.py` (new).
