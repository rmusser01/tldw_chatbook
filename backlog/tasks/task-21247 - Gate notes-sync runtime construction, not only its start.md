---
id: TASK-21247
title: >-
  Gate notes-sync runtime construction, not only its start
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - notes-sync
  - startup
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; finding from the
TASK-21108 review.

TASK-21112 gated the notes-sync runtime's `start()` on real evidence of configuration. Its own
evidence shows the remaining cost is in **construction**, not start.

On dev `b2b1e2e0d`, `TldwCli.__init__` builds the owner unconditionally
(`app.py:6032` `build_notes_sync_runtime_owner(...)`, with the module-level import at `:379`),
so the ~15-module chain is paid at boot by every user — including one with zero profiles
configured, the exact case TASK-21112 was written for.

On the in-flight TASK-21108 branch (`origin/fix/task-21108-wave5`, **not merged into dev** at
close-out) that construction moves behind a lazy property, but `on_mount` reads the property
unconditionally — so the chain is relocated to just before first paint rather than removed.

Either way, the same `start_evidence` predicate that already decides whether to start could
decide whether to construct. Re-confirm the exact call shape against dev after TASK-21108
merges.

## Acceptance Criteria

- [ ] A boot with no notes-sync configuration constructs no notes-sync runtime owner and imports none of its dependency chain
- [ ] A configured profile, and the one-time legacy `[notes]` sync-directory migration path, behave exactly as they do today
- [ ] The gate predicate stays side-effect free — evaluating it never creates or opens the state store
- [ ] An import-closure guard names the notes-sync chain and fails if a zero-profile boot pulls it in

## Implementation Plan

1. Re-confirm against dev (TASK-21108 wave5 has landed): construction is a
   locked lazy property, but `on_mount` reads it unconditionally -- premise
   confirmed by the `_ui_ready` census (10 `notes_sync_*` modules resident).
2. Extract `_notes_sync_boot_start_evidence()` (the TASK-21112 gate plus the
   TASK-21240 empty-store refinement) and evaluate it in `on_mount` BEFORE
   touching the property; fail open on probe error, matching `_start_once`.
3. Pin the win: add the five construction-chain modules to the census's
   `ABSENT_AT_READY_MODULES` (the census boots the real app headless --
   the zero-profile boot the AC names).
4. Unit tests for the predicate and the fail-open wrapper.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Fix**: `app.py` `on_mount` now evaluates
  `_notes_sync_boot_start_evidence(self)` (wrapped `except Exception: True`)
  and only reads `self.notes_sync_runtime_owner.start()` when evidence
  exists; otherwise it logs and leaves the slot unconstructed (first-time
  setup still force-constructs on demand via `review_setup`). Shutdown was
  already safe: `_shutdown_notes_sync_runtime` reads the private slot
  first and skips when None.
- **Measured**: `_ui_ready` census warm boot 1033 -> **1028**; the
  construction chain (`notes_sync_authority/coordinator/executor/runtime/
  watcher`) went resident-absent. Still resident, honestly attributed in
  the census comments: `notes_sync_legacy` (the gate predicate itself
  imports it) + `notes_sync_filesystem` (pulled by legacy), and
  `conflicts`/`models`/`reconciler` (module-level imports of the Library
  notes-sync controller -- pre-existing boot cost, not construction).
- **Guard**: `Tests/Performance/test_ui_ready_module_census.py`
  `ABSENT_AT_READY_MODULES` now names the five construction-chain modules
  (4 passed, including the full 2-boot subprocess census). Unit coverage in
  `Tests/App/test_notes_sync_boot_gate.py`.
- **No regression**: `Tests/Notes/test_notes_sync_cutover.py` shows the
  same 2 pre-existing local failures as HEAD (shared-profile
  RecoveryRequired class, verified by checkout swap);
  `Tests/App/test_boot_no_feature_db_files.py` green.
