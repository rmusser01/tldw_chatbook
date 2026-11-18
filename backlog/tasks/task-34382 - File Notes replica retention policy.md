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

## Qodo review round (PR #3016) — 2026-10-06

Disposition of the retention findings against this task's code
(`enforce_retention` + invocation seams).

- **HIGH 1 — the checkpoint cap could evict the NEWEST checkpoint.** Fixed.
  Rule 1 chose its 50 survivors with SQL `ORDER BY created_at DESC` — text
  order, not time order, for the mixed `Z`/`+00:00`/offset spellings the
  replica stores (the module's own docstring already promised parsed-UTC
  comparison; Rule 1 was the one place still sorting in SQL). Survivors are
  now ranked in Python by `_revision_row_rank` (parsed UTC instant, rowid
  tie-break); unparseable stamps rank newest and are kept (ADR-218
  fail-safe). Red pin (pre-fix):
  `test_checkpoint_cap_ranks_mixed_timestamp_spellings_by_instant` evicted
  exactly the newest recovery copy whose `-09:00` spelling sorts lexically
  lowest.
- **M6 — tied greatest `deleted_at` exempted every peer.** Fixed. Rule 2
  now preserves exactly ONE most-recent tombstone, chosen by
  `(parsed instant, files rowid)`; tied peers expire. Red pin (pre-fix):
  `test_tied_expired_tombstones_keep_exactly_one` kept both (2 != 1).
- **M7 — a retained tombstone could lose its bytes.** Fixed. Rule 3 exempted
  a delete revision by `created_at == newest deleted_at`, but
  `prepare_deletion` records the two stamps independently. The preserved
  tombstone's delete revision is now identified by row identity (the newest
  `delete` row of the preserved path — the same row a NULL-session lookup
  serves), never by timestamp equality. Red pin (pre-fix):
  `test_preserved_tombstone_keeps_its_delete_revision_when_stamps_differ`.
- **M14 — protected notes never hit the cap.** Fixed (policy corrected).
  Only protected saves write `pre_edit` checkpoints, so Rule 1's protected
  exemption left exactly the ADR-218 growth scenario unbounded and
  contradicted the ADR's own Consequences ("up to 50 sessions"). The cap
  now applies to every path; protected paths stay exempt from the 30-day
  expiry only. ADR-218 Decision items 1/3/4 and the rejected-alternative
  bullet were corrected to match the Consequences. Pins: updated
  `test_checkpoint_cap_evicts_the_oldest_beyond_the_bound` (kept.md now
  capped) and new
  `test_protected_checkpoints_are_capped_but_never_expire`.
- **M10 — shutdown retention failures went unreported.** Fixed. `shutdown()`
  now inspects the returned `OperationResult` and logs the message/status
  when the sweep did not succeed (exceptions were already logged); the
  seam stays non-blocking. Both shutdown log paths use the module-level
  loguru logger: the first cut used the DOM `self.log` and regressed 16
  workspace tests that call `shutdown()` after the app exits
  (`NoActiveAppError` — the DOM logger needs the `active_app` context var);
  recorded in `backlog/docs/lessons-textual.md` and caught by the
  workspace-battery A/B below before anything was pushed.
- **M11 — retention `fetchall()` collections.** Rejected with
  justification. The three collections are metadata-only tuples (no BLOB
  is ever loaded in retention) and are bounded by the schema and the
  policy itself: tombstones ≤ one per path (`UNIQUE(root, relative_path)`),
  checkpoints ≤ cap + ε per path after this round, and the tombstone pass
  is an inherently full-scan reduction (the single most-recent tombstone
  must be known before anything is evicted, so batching cannot bound it).
  Collect-then-delete is also the SQLite-safe pattern — modifying a table
  while a SELECT cursor iterates it is undefined behavior — and ADR-218
  deliberately runs the three rules in one transaction. Keyset batching
  would add failure modes to a correctness-critical transaction without a
  measurable memory win; the durable answer to pathological volume is the
  retention policy itself, not pagination inside its sweep.

Evidence: `python -m pytest Tests/Notes/test_file_notes_replica.py
Tests/Notes/test_file_notes_retention.py
Tests/Notes/test_file_notes_service.py
Tests/Notes/test_file_notes_delete_safety.py
Tests/Notes/test_file_notes_session_owner.py
Tests/UI/test_library_file_notes_history.py
Tests/UI/test_library_file_notes_delete_safety.py
Tests/UI/test_library_file_notes_retention.py -q` → **206 passed** with
this round's changes (new red pins verified failing against pre-fix HEAD
first, via `git show HEAD:` file swaps — never `git stash`).

Regression A/B, `Tests/UI/test_library_file_notes_workspace.py`
(`--tb=no`, sequential same-machine runs): HEAD baseline **54 FAILED**, all
pre-existing (the config-admission `RecoveryRequired` reds recorded in the
pre-review notes). The first fixed run showed **70 FAILED** — the diff
isolated 16 names to this round's `self.log` shutdown regression (fixed as
above; bisected by single-file swap, red pin `NoActiveAppError`). The
final full-fix run's FAILED list is **byte-identical to the HEAD baseline**
(54 names, zero new, zero gone). One intermediate post-fix run additionally
showed `test_initial_root_scan_projects_checking_authority_while_actions_a
re_gated`, a ~1-in-6 paint-timing flake whose failure mode is a middle-
elided `#file-notes-root-status` string (`'Chec…otes'`) — reproduced 1-in-6
on the fixed tree AND 0-in-4 at HEAD in calm single runs, with no code-path
overlap (this round touches neither root-status painting nor the scan
gating), i.e. the same pre-existing `_wait_until` flake family the
pre-review notes already record for this battery; it did not appear in the
final A/B run.
