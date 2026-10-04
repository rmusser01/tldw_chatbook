---
id: TASK-34382
title: >-
  File Notes replica retention policy: checkpoint cap, tombstone and revisions expiry
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
reconnaissance). Without retention the replica grows unboundedly: checkpoints accumulate per session forever and tombstones/revisions never expire. This task adds bounded retention inside `file_notes_replica.py`.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Per-note checkpoint cap (order ~50) evicting oldest beyond the cap
- [x] Tombstone and revision expiry (~30 days) with cleanup invoked on root change and/or session end
- [x] Protected paths and the most recent tombstone are never evicted by the policy
- [x] Retention is test-pinned (counts before/after, protected-preservation, recency)
<!-- AC:END -->

## Implementation Plan (to be added when claimed)

ADR required: yes (amendment-class storage policy inside ADR-029's replica)
ADR path: backlog/decisions/218-file-notes-replica-retention.md
Reason: retention (checkpoint cap ~50, tombstone/revision expiry ~30 days,
protected-path and newest-tombstone preservation) is a storage-policy decision
about the ADR-029 replica's data lifetime — AGENTS.md requires an ADR for
storage/conflict-policy decisions. Numbers ≤217 are taken on dev (ceiling is
ADR-217, so the brief's "209+" range was re-derived at filing time per the
backlog-hygiene lesson); 218 is free on dev and in this worktree. ADR-029
itself is not superseded — this amends its "Consequences" deferral of
quotas/retention with concrete bounds.

1. Replica (`file_notes_replica.py`): constants `MAX_REVISIONS_PER_NOTE = 50`,
   `RECOVERY_EXPIRY_DAYS = 30`; `enforce_retention(root, *, now)` in ONE
   transaction — per-note `pre_edit` cap (evict oldest beyond cap, protected
   paths exempt), tombstone expiry (delete tombstoned `files` rows + FTS past
   the cutoff, the most-recent tombstone always kept), revision expiry (past
   cutoff, protected paths and the most-recent tombstone's delete revision
   exempt). Timestamps parsed in Python (mixed `Z`/`+00:00` spellings);
   unparseable values fail safe (kept).
2. Service (`file_notes_service.py`): `_serialized enforce_retention()` bound
   to `self.root_key`, returning a typed `OperationResult`.
3. Invocation (`library_file_notes_workspace.py`): `set_root` enforces after a
   successful scan and BEFORE loading Recently-deleted (so the list reflects
   cleanup); `shutdown()` enforces after the save task settles, before owner
   shutdown. Both guarded — retention failure logs, never blocks.
4. Tests (red first): `Tests/Notes/test_file_notes_retention.py` (cap counts
   before/after, protected preservation, recency of survivors, newest-tombstone
   preservation, lone-old-tombstone kept, service-level enforcement) and
   `Tests/UI/test_library_file_notes_retention.py` (set_root and shutdown each
   invoke the service seam, via `_WorkspaceHarness`).


## Implementation Notes

ADR-218 (`backlog/decisions/218-file-notes-replica-retention.md`) created before
implementation — retention is a storage-policy decision inside the ADR-029
replica; ADR-029 is amended, not superseded. Index row added to
`backlog/decisions/README.md`. Numbering: dev's shipped ceiling is ADR-217, so
the brief's "209+" range was re-derived at filing time; 218 is free on dev.

**Approach.**
- `file_notes_replica.py`: constants `MAX_REVISIONS_PER_NOTE = 50`,
  `RECOVERY_EXPIRY_DAYS = 30`; `enforce_retention(root, *, now)` applies three
  rules in ONE transaction — (1) per-note `pre_edit` cap keeping the 50 newest
  by `(created_at, session_key)`, protected paths exempt; (2) tombstone
  expiry: tombstoned `files` rows (plus their FTS rows) past the cutoff are
  dropped, but the most-recent tombstone in the root never is; (3) revision
  expiry past the cutoff, exempting protected paths and the preserved
  tombstone's own `delete` revision (the same recovery fact). Timestamps are
  parsed in Python (`_parse_utc_timestamp`: `Z`/`+00:00` both occur in stored
  rows, so SQL string ordering is unsafe) and unparseable values fail safe
  (kept). `is_protected` was refactored onto a shared
  `_protected_row_exists(cursor/connection, ...)` so retention reuses the
  exact protection-matching SQL inside its transaction.
- `file_notes_service.py`: `_serialized enforce_retention()` bound to
  `self.root_key` returning a typed `OperationResult`
  (ok / replica-error).
- `library_file_notes_workspace.py`: `set_root` enforces after the candidate
  scan and BEFORE `_load_deleted_paths` (the Recently-deleted list never
  names entries just expired); failure merges into the scan's
  `replica_warning` and never blocks the change. `shutdown()` enforces after
  the in-flight save settles and before owner/replica retirement; failure
  logs via `self.log.warning`. New `_enforce_retention(service, generation)`
  helper owns the thread hop + staleness check.

**Red → green evidence** (worktree `.worktrees/wave5-build`, Python 3.12.13).
- Red: `python -m pytest Tests/Notes/test_file_notes_retention.py -q` →
  `ImportError: cannot import name 'MAX_REVISIONS_PER_NOTE'` (constants/API
  absent). Invocation seam red: with the workspace file reverted to HEAD
  (`git checkout HEAD -- <workspace.py>`, service changes kept),
  `python -m pytest Tests/UI/test_library_file_notes_retention.py -q` →
  `FAILED … AssertionError: assert 0 >= 1` (workspace never invoked the
  seam).
- Green: `python -m pytest Tests/Notes/test_file_notes_retention.py
  Tests/UI/test_library_file_notes_retention.py -q` → `9 passed`;
  combined file-notes batteries
  (`test_file_notes_replica.py test_file_notes_service.py
  test_file_notes_retention.py test_library_file_notes_history.py
  test_library_file_notes_retention.py`) → `85 passed`.
- Regression A/B: `python -m pytest Tests/UI/test_library_file_notes_workspace.py
  Tests/UI/test_library_notes_w4_file_notes.py
  Tests/UI/test_library_notes_riders_r_file_notes.py
  Tests/UI/test_library_notes_wave_file_notes.py -q` FAILED list (76 names)
  is byte-identical to the pre-34381 baseline (same pre-existing
  config-admission `RecoveryRequired` reds); set_root/shutdown changes
  introduced no new failures.

**Boundary semantics pinned.** "30 days" evicts strictly older than the
cutoff; exactly-30-days survives (`test_expiry_boundary_keeps_exactly_thirty_days_and_evicts_past_it`).

**Files changed.** `tldw_chatbook/Notes/file_notes_replica.py`,
`tldw_chatbook/Notes/file_notes_service.py`,
`tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`,
`backlog/decisions/218-file-notes-replica-retention.md` (new),
`backlog/decisions/README.md`,
`Tests/Notes/test_file_notes_retention.py` (new),
`Tests/UI/test_library_file_notes_retention.py` (new).
