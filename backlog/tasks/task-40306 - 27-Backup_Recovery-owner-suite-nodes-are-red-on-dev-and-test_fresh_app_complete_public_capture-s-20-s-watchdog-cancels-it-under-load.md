---
id: TASK-40306
title: "27 Backup_Recovery owner-suite nodes are red on dev, and test_fresh_app_complete_public_capture's 20 s watchdog cancels it under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 06:26
labels:
- follow-up
- tests
- backup
- recovery
- dev-red
priority: medium
references:
- Tests/Backup_Recovery/test_created_persona_subtree_rollback.py
- Tests/Backup_Recovery/test_restore_plan.py
- Tests/Backup_Recovery/test_visual_identity_restore.py
- Tests/Backup_Recovery/test_inventory_unused_owners.py
- Tests/Backup_Recovery/test_builtin_later_snapshot.py
dependencies:
- TASK-39116
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (census stream review r13b-rev1; plugins stream, TASK-40132). Measured red at base ae9f9e8d21 and at the census branch head; no round-13 commit touches these files except a test that moved out of `test_restore_plan.py` (net zero). Checked by reading at 8b633d0101, 2026-11-12.

**Why.** These files run on no PR lane, so dev went red in them unseen, and every covering run has to subtract the same 27 by hand, which is how a new regression hides.

**The reds** (all under `Tests/Backup_Recovery/`):
- `test_created_persona_subtree_rollback.py`: the module fixture `native_persona_negatives` (`:421`) feeds the parametrized `test_native_persona_mapping_refuses_unproved_graph` (`:441`): 19 errors and 1 failure.
- `test_restore_plan.py:790-791`: `test_shipped76_archive_stages_installed_schema_without_changing_source[False|True]` (TASK-40132 also saw both red at its base 42fc21939d).
- `test_visual_identity_restore.py` (1), `test_inventory_unused_owners.py` (3) and `test_builtin_later_snapshot.py` (1).

**The flaky watchdog.** `test_inventory_unused_owners.py:163` (inside the `_FRESH` child script, from b5251e9a6e) sets `call_later(20, cancel.set)`. `test_fresh_app_complete_public_capture` (`:192`) then fails with `InterruptedError: cancelled` under xdist load (two TASK-40132 covering runs), and passes alone in about 22 s. A watchdog shorter than a loaded run reports a cancellation where nothing hangs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every node of the five files passes at dev head at `-n 4`, each red first given a named cause (test drift with the commit, or a product defect fixed with a test that was red)
- [x] #2 test_fresh_app_complete_public_capture's watchdog fires only on a real hang (derived from or replaced by the outer subprocess timeout), and the file passes 3 of 3 at `-n 4` under load
<!-- AC:END -->

## Implementation Plan

1. Show every red node of the five files at base 26b2e677dd (-n 4) and record each failure's message.
2. For each red, find the root cause (git log -S/-L, and a run of the node on a git-archive tree of the commit that added it where the history is unclear): test drift named with its commit, or a product defect.
3. Fix test drift in the tests; fix product defects in the product with a test that was red first.
4. test_fresh_app_complete_public_capture: replace the fixed 20 s cancel with one derived from the child's hang budget, and turn the 500 x 10 ms poll into a condition wait bounded by the same deadline.
5. Rerun the five files at -n 4 (load recorded) until every node passes; run the inventory file 3 of 3 under load.

## Implementation Notes

Base 26b2e677dd, five files at `-n 4` (load 70-100): 10 failed, 19 errors (29; the filed 27 plus two more `test_builtin_later_snapshot` timeouts at that load). Each red, its cause and fix:

- **`test_restore_plan.py::test_shipped76_archive_stages_installed_schema_without_changing_source[False|True]`: test drift.** 7970fbcada (2026-11-01, prompt mix-ins) added `mixin_snapshot_schema_version` to `console_dispatch_checkpoints`; restored v76 receipts carry it as NULL, and the test popped only the earlier added column. It now pops every column added after its v76 fixture and requires each to be NULL.
- **`test_inventory_unused_owners.py::test_installed_owner_inventory_contract[graph_referenced]`: drift, red since the test landed.** Legacy Collections are read-only since 534f0cb899 (`create_collection` raises `LegacyCollectionsReadOnlyError`), an ancestor of b5251e9a6e, the commit that added the test. The child now seeds a retained membership by SQL, as `Tests/Library/test_collections_legacy_recovery.py` does.
- **`test_inventory_unused_owners.py::test_exact_lazy_defaults_preserve_required_missing_evidence`: wrong expectation, red since the test landed.** `notes.sync_state` is a device-local store that backup excludes; `_excluded_root` (b5251e9a6e) reports an absent excluded root as `intentionally_excluded`, never `unused`. The test asserts that for it and keeps the five lazy owners.
- **`test_visual_identity_restore.py`: drift, red since the test landed.** Its config named a custom `research_db_path` that nothing created: since 92f9dba52c (TASK-21105) the research DB opens on first use, and an explicit selector whose file is missing is `missing_required` by design (`_sqlite_inventory_status`). The fixture no longer names a research path; the test is about visual identity.
  - "Red since landing" for these three was checked by running them on a git-archive tree of b5251e9a6e: same three errors.
- **`test_fresh_app_complete_public_capture_preserves_real_bytes_and_resumes`: fixed watchdog (AC #2).** At base it failed with `ValueError: cancelled` (20 s `call_later`), and with the old 45 s child limit it would fail anyway: the child took 62-72 s at load ~50. The child's cancel now fires 15 s before a 240 s hang budget it gets from the parent (`FRESH_CAPTURE_DEADLINE`), and the 500 x 10 ms pause poll is a condition wait bounded by that cancel.
- **`test_created_persona_subtree_rollback.py` (19 errors, 1 failure): slow children, not hung.** The module fixture's replacement child runs two whole-profile replaces (the credential review aborts the first), and every xdist worker builds the fixture, so at load 40-100 it outlasted its fixed 150 s. At base the restore journal's quadratic cost (TASK-40310) made it slower still. With TASK-40310 it still took >150 s at load ~40-50, and passed with the budget lifted (fixture setup 283 s). Fix: the replacement and later-rollback children now fail only once their process tree uses no CPU for 60 s (`_wait_while_busy` in `test_config_sibling_capture.py`, opt-in `_child(..., idle=)`), under a cap. `test_child_wait_fails_only_an_idle_child` pins it; with the idle check disabled by a temporary edit its idle case fails `DID NOT RAISE`.
- **`test_builtin_later_snapshot.py`: load.** `test_fresh_process_rederives_finite_sources_and_executes_later_rollback`'s child repeats a later rollback that took 87 s in-process at load 50 against a fixed 60 s; it now uses the same idle wait. The two `test_shared_builtin_later_preview_and_execution_preserve_both_profile_trees` cases ran into pytest's 300 s default (setup 107-135 s plus call 122-172 s at load 36-57); unchanged, they passed in later runs.

AC #1: every node of the five files passed at `-n 4` in TASK-40307's lane runs (all lane files together): run 1, 509 passed at load 26-57; run 2, 509 passed at load 29-51.

Still seen, not fixed (proposed follow-up): in one `-n 4` run at load 36-57, one of the five persona fixture builds failed in its reopen child with `sqlite3.OperationalError: database is locked` on `add_note` right after the app mounted (the test's `observe_notes_write_failure` record was lost with the basetemp). It did not recur in the lane runs below.

AC #2: `test_inventory_unused_owners.py` passed 3 of 3 at `-n 4`: in the five-file run (load 46-55) and twice alone at load ~50 (fresh node 72 s and 62 s).

Lesson added to `backlog/docs/lessons-testing-evidence.md` (fixed child budgets under load; xdist builds a module fixture once per worker).

Files: `Tests/Backup_Recovery/test_restore_plan.py`, `test_inventory_unused_owners.py`, `test_visual_identity_restore.py`, `test_config_sibling_capture.py`, `test_created_persona_subtree_rollback.py`, `test_builtin_later_snapshot.py`.
