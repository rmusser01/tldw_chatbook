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

## Qodo review round (PR #3016) — 2026-10-06

Disposition of the review findings against this task's read-path code
(listing, verify, export, restore). All dispositions verified against the
code before acting; red pins were confirmed against the pre-fix HEAD
(`git show HEAD:` file swap, no stash) before the fixes landed.

- **HIGH 2 — deleted-copy export could write an older deletion's bytes.**
  Fixed. `get_revision` served an arbitrary NULL-session `delete` row
  (fetchone without ordering; NULLs are distinct under the UNIQUE index, so
  every delete cycle inserts another row). Delete revisions now carry a
  stable `revision_id` (rowid) surfaced through `ReplicaRevisionInfo` and
  the service's verify/export/restore; identity-less NULL-session lookups
  serve the NEWEST deletion by parsed instant (the current tombstone's own
  snapshot). Red pin (pre-fix): `test_each_deletion_cycle_serves_its_own_bytes`
  served the FIRST cycle's bytes; service-level
  `test_deleted_copy_export_writes_the_current_tombstones_bytes` and
  `test_history_revision_ids_select_each_deletion_exactly` pin the
  delete→restore→re-delete sequence end to end.
- **M3 — unbounded history page size.** Fixed. `REVISION_HISTORY_MAX_LIMIT
  = 200` is clamped in both the service (`list_revision_history`) and the
  replica (`list_revisions`). Pin:
  `test_history_page_size_is_clamped_to_the_shared_ceiling` (red pre-fix:
  the constant did not exist) and
  `test_revision_history_limit_is_clamped_to_the_shared_ceiling`.
- **M5 — two independent `10` defaults.** Fixed. The default lives once in
  the replica module (`REVISION_HISTORY_DEFAULT_LIMIT`); the service's
  `REVISION_HISTORY_LIMIT` aliases it.
- **M12 — bounded history could omit the newest revision.** Fixed.
  `list_revisions` no longer sorts `created_at` as SQL text; rows rank by
  `_revision_row_rank` (parsed UTC instant, rowid tie-break), and the
  listing reads `length(raw_bytes)` instead of loading the blobs. Red pin
  (pre-fix): `test_list_revisions_orders_by_parsed_instant_not_text`
  returned the older revision at the limit boundary.
- **M4 — History dialog destination skipped shared validation.** Fixed.
  `FileNotesHistoryDialog._destination()` applies the same
  `validate_text_input(..., max_length=4096, allow_html=True)` the
  `#file-notes-path` input gets; service path-safety checks unchanged. UI
  pin: `test_history_destination_input_rejects_unsupported_text` (red
  pre-fix: the dialog dismissed with the request).
- **M13 — history actions could act on another root.** Fixed. Every
  `_HistoryActionRequest` carries the `service` and `root_generation` that
  supplied the dialog's listing plus the `revision_id`; execution refuses
  when either identity changed, and `_show_history` re-checks the opened
  file after its await. UI pin:
  `test_history_action_is_rejected_after_the_root_changes` (red pre-fix:
  the type did not carry the identity).
- **M9 — export could report failure after publishing.** Fixed. After the
  `os.link` publishes verified bytes, a parent-fsync failure is a
  post-publication durability warning: the destination is recorded
  (`SessionChange("created")`), the result is `ok` with
  `replica_warning`, and the workspace receipts append the warning.
  Retrying no longer meets a bogus `exists` for an unrecorded file. Pin:
  `test_revision_export_reports_durability_warning_after_publishing`
  (injected parent-fsync OSError; red pre-fix: status `error` with the
  file published but unrecorded). The parallel pre-existing shape in
  `export_exact_file` (task-32896, outside this PR) was left untouched.

Evidence: `Tests/Notes/test_file_notes_replica.py`
`Tests/Notes/test_file_notes_service.py`
`Tests/UI/test_library_file_notes_history.py` green after the fixes;
full-battery A/B below in 34382's section.
