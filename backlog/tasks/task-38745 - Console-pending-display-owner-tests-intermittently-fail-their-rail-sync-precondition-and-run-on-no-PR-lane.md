---
id: TASK-38745
title: "Console pending-display owner tests intermittently fail their rail-sync precondition and run on no PR lane"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-04 06:34
updated_date: 2026-11-05 22:05
labels:
- follow-up
- tests
- console
- flaky
priority: medium
references:
- Tests/UI/test_console_pending_display_owner_controls.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/UI/test_console_pending_display_owner_controls.py` has two tests, both `@private_profile_test`:
- `test_pending_fragment_current_owner_and_saved_generation_remain_display_only` (:48);
- `test_pending_fragment_declines_unknown_owner_custom_reader_and_malformed_metadata` (:84).

Both start with the shared `_prepared` helper (:30). It asserts `await projection.warm()` (:33), then `console._sync_console_rail_and_controls() is not False` (:34).

In large `-n 4` runs during round 6, both nodes failed now and then at :34: the warm-up had returned True, and the rail sync still returned False. TASK-38650's notes also record a dev failure in this file. On this base, a single `-n 4` run of this file together with two other files failed the first node at :34 (`assert False is not False`). The load average was about 8 on 18 CPUs.

`_sync_console_rail_and_controls` (`UI/Screens/chat_screen.py:20945`) returns the result of `run_console_presentation_sync`, and False means that sync deferred. So the precondition depends on timing that a successful warm-up does not settle. The helper does not say which deferral it expects to have cleared.

The file runs on no PR lane. It has no entry in `scripts/ui_pr_gate_census.txt` and is not a target in any workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes name, from evidence, which deferral makes the rail sync return False after a successful warm-up
- [x] #2 `_prepared` waits for the state its later asserts depend on, or, if the deferral is a product bug, the product stops deferring there; the failure is reproduced before the change, under load or with a temporary source edit
- [x] #3 Both nodes pass 20 of 20 `-n 4` runs of the file, with the load average recorded in the notes
- [x] #4 The file runs on a PR lane (a census entry or a workflow target), and its measured time in that lane is recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe warm-up + rail-sync pairs under load and record which branch of the checked display refused each False.
2. `_prepared` retries warm-up + sync until the sync renders (bounded), instead of asserting the first sync.
3. Reproduce each observed deferral against the old helper with a temporary test edit; show the new helper passes the same edit.
4. 20 `-n 4` runs of the file; append the file at the census tail and record its time in that lane.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The deferral (AC#1).** A temporary probe seeded the Console the way `_prepared` does, then repeated warm-up and rail sync up to 40 times per app. It wrapped `ConsoleReadinessConfigProjection.run`, `_run_checked_display_sync`, `_checked_display_status` and `run_console_config_sync`. In 5 probe batches (556 warm/sync pairs, load 49-68), 6 syncs returned False right after `warm()` had returned True. Every one was the checked readiness display refusing in `_run_checked_display_sync`, with both config locks free and no maintenance pause. The refusing branch was recorded for the last 5:
- **4 of 5: the display proof expired during the sync.** `warm()` returns True for a value younger than `max_age` (1 s), so it can hand back one 0.978-0.996 s old without refreshing it. `_checked_display_status`'s age check (`time.monotonic() - proof.at >= min(1.0, max_age)`) then refused at 1.004-1.063 s.
- **1 of 5: the storage-admission lock was busy.** At a proof age of 0.31 s, another thread held `storage_admission._lock` (an RLock), so `_checked_display_status`'s nonblocking acquire failed. The holding thread was not captured.

The first False came before branch recording was added: its proof was 0.87 s old when the sync started.

Neither is a product bug. Both are the checked display deferring to its own retry, as designed. The first sync after the seed did not fail in the 40 probes that ran one; all six falses came on later pairs.

**Change (AC#2).** `_prepared` in `Tests/UI/test_console_pending_display_owner_controls.py` now repeats warm-up and rail sync until a sync renders, bounded at 30 s, with a 0.05 s pause between tries.

**Reproduced before the change, with temporary test edits.** Both nodes failed at the old line 34 with `assert False is not False` under each edit, and both passed with the new helper under the same edit (applied to its first try only):
- sleeping until the proof is 1.0 s old between `warm()` and the sync (old helper: load 70; new helper: load 63);
- holding `storage_admission._lock` from another thread for 0.5 s before the sync (old helper: load 50; new helper: load 45).

**AC#3.** 20 of 20 `-n 4` runs of the file, 2 of 2 nodes each, at load 22.5-40.7 (1-minute average).

**AC#4.** Whole-file entry appended at the tail of `scripts/ui_pr_gate_census.txt`, with `MINIMUM_FILES` raised to the measured count in `scripts/check_ui_pr_gate_census.py`. The checker and `Tests/CI/test_ui_pr_gate_census_checker.py`, `test_ui_lane_shard_count_statements.py` and `test_ui_lane_shard_replay.py` pass (115). Time with the lane's serial invocation (`--timeout=180 --durations=0`, no xdist), locally: 20.0 s, 26.0 s and 21.9 s for the file, with calls of 8.8-12.7 s per node (load 10-13). It has not been timed on the Linux CI runner.
<!-- SECTION:NOTES:END -->
