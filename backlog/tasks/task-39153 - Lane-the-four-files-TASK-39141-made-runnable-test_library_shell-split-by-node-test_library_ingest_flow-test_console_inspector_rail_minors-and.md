---
id: TASK-39153
title: "Lane the four files TASK-39141 made runnable: test_library_shell split by node, test_library_ingest_flow, test_console_inspector_rail_minors and test_persistent_terminal_privacy_boundary"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-10 20:24'
created_date: 2026-11-08 06:54
labels:
- follow-up
- ci
- tests
- library
priority: medium
references:
- Tests/UI/test_library_shell.py
- Tests/integration/test_library_ingest_flow.py
- Tests/UI/test_console_inspector_rail_minors.py
- Tests/Architecture/test_persistent_terminal_privacy_boundary.py
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-39141
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39141's timings, round-12 branch fu/r12-ctlr). Checked on the round-12 stack 9091820bf2, 2026-11-08: none of the four files is in `scripts/ui_pr_gate_census.txt` or in any workflow.

**Timings** (from TASK-39141's notes: run.sh, `-n 4`, the Architecture file serial):
- `test_library_shell.py`, 848 nodes:
  - 992 s on base at load 45-49, with 785 nodes failing fast;
  - 2,073 s on head at load 45-160, and 5,267 s at load ~100;
  - 1,302-1,365 s for three of the files together at load 33-57.

  TASK-39141's notes put one UI census entry at about 30 s, so this file has to be split by node across shards. Its stale nodes are re-pinned first (TASK-39152).
- `test_library_ingest_flow.py`: 17-27 s, and 63 s at load ~100.
- `test_console_inspector_rail_minors.py`: 12-24 s, and 51 s at load ~100 (5 passed on head).
- `test_persistent_terminal_privacy_boundary.py`: 42-121 s, and 151 s at load ~100 (8 passed on head). It runs serially, so it belongs in the Architecture guards job.

**The UI lane.** The census has no per-entry budget; its rule is per shard. On the round-12 stack the UI lane runs 36 shards (TASK-39144), and the census header (`scripts/ui_pr_gate_census.txt`) keeps each job's slowest run 5 min under its 20-min cap, checked with `scripts/ui_lane_shard_replay.py`. The census's other rules and floor are in `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_console_inspector_rail_minors.py` and `test_library_ingest_flow.py` run on a PR lane, with three timings of each lane step or shard recorded
- [x] #2 `test_persistent_terminal_privacy_boundary.py` runs in the Architecture guards job, with three timings of that step recorded
- [x] #3 `Tests/UI/test_library_shell.py` runs, split by node, in a job that fires on pushes to dev (not on PRs), with each shard's time recorded (reworded 2026-11-10 by owner decision; it read: "`test_library_shell.py` runs on PR lanes, split by node so that `scripts/ui_lane_shard_replay.py` shows every UI shard's slowest replayed run still at least 5 min under the 20-min cap, with three timings of each changed shard recorded")
- [x] #4 None of the four files runs on the fast PR contract
<!-- AC:END -->

## Implementation Plan

1. AC#2: append `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` at the end of the Architecture guard step (its bootstrap node after every sandboxed suite) and to its pin in `Tests/CI/test_ci_queue_pressure_contract.py`; time the step's own command serially 3 times.
2. AC#1: `test_library_ingest_flow.py` (Tests/integration, so not census-eligible) joins the recovered-suites step after its bootstrap_profile suites, guard-clean under `--storage-lease-guard`; `test_console_inspector_rail_minors.py` joins the UI census tail; time each 3 times.
3. AC#3: once TASK-39152's re-pins land, price `test_library_shell.py` by node from junit durations and compute the 36-shard floor before any re-deal (lessons-testing-evidence, TASK-36512): if the floor is over 13.3 pytest min, no order meets it and the shard count is an owner decision; otherwise append its green nodes at the census tail and run `scripts/ui_lane_shard_replay.py --rebalance 13.3` over recent 36-shard logs.
4. AC#4: none of the four files joins `pr-fast-lane`.

## Implementation Notes

**Status: all four files run on a lane. Three run on PR lanes (AC#1, AC#2, AC#4); `test_library_shell.py` runs by node in a job on pushes to dev (AC#3, reworded by the owner decision below).**

**Owner decision (2026-11-10, AskUserQuestion, relayed by the coordinator), verbatim:** "Push-to-dev job only (Recommended)" -- option text: "Run it split by node in a job that fires on pushes to dev (and nightly), not on PRs. No extra PR runner cost; a regression shows up within one merge. AC#3 is reworded from 'PR lane' to this, with your decision recorded." Implemented as push-to-dev only: a `schedule` trigger only runs the default branch's (main's) copy of the workflow file, so "nightly" cannot apply to dev's workflow.

**AC#1.**
- `Tests/UI/test_console_inspector_rail_minors.py` is a whole-file entry at the UI census tail (bootstrap_profile, 5 tests; committed with TASK-39130's five nodes, floor 1913 -> 1919). Three serial runs with the lane's flags (run.sh, `--timeout=180`, load 72-88): 92.0 s, 72.7 s and 94.0 s of pytest, 5 passed each. `scripts/ui_lane_shard_replay.py --shards 36` over the 22 saved round-12 PR runs, every new entry charged at its slowest local time: max shard 12.3 -> 13.1 pytest min, under the header's 13.3.
- `Tests/integration/test_library_ingest_flow.py` is not census-eligible (the checker takes `Tests/UI/` only), so it joins the recovered-suites step (`PR lane (recovered suites)`), last, after its bootstrap_profile suites, under `--storage-lease-guard`; its pin in `Tests/CI/test_ci_queue_pressure_contract.py` follows. The file alone, serial, guard on: 13 passed in 55 s (load 31-34). The whole step's own command, serial, three times: 839 s, 915 s and 668 s of pytest (21 targets, 746 passed and 1 skipped each, load 60 -> 35). That job took 6.1-6.5 min in CI before (runs 42841528703, 42841511400) against its 20-min cap.

**AC#2.** `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` runs last in the Architecture guard step (its one bootstrap_profile node after every sandboxed suite) and in `ARCHITECTURE_GUARD_TARGETS`. The file alone, three serial runs: 50.7 s, 56.2 s and 46.0 s, 8 passed each (load 33-38). The whole step's own command (36 targets), serial, three times: 1,324 s (load 84 -> 46), 1,045 s (load 46 -> 38) and 591 s (load 38 -> 28). The new file passed in all three. Runs 1 and 2 each had one unrelated failure, `test_profile_owned_path_inventory.py::test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners` hitting the 180 s timeout under load (180.0 s call). That test runs before the new file, and run 3 passed all 441. The job took 7.1-10.3 min in CI against its 20-min cap.

**Why not a PR lane: the floor (lessons-testing-evidence, TASK-36512: compute it before any deal).**
- The current census replays to 420.6 pytest min over 36 shards after this round's six entries (max 13.1), so at most 36 x 13.3 - 420.6 = 58.2 min of room is left, and only under perfect packing.
- `test_library_shell.py` (848 nodes): its junit durations summed to 8,683 s in the base run at `-n 4` (load 70-100). A serial run of 40 random green nodes took 1.066 times their `-n 4` time (load 64-96), so the file is about 154 min serial locally.
- TASK-39144's lesson measured CI at 0.51-0.99 of local serial time (median 0.73). That puts the file at 79-153 min in CI, median 113 min, or 13.9-15.9 min per shard over 36, against 13.3. No order meets it; the lane needs about 38-44 shards (40 at the median), or a job of its own. Both are owner decisions (the 36-shard count is the owner's 2026-11-07 decision), so no `test_library_shell.py` node is in the census.
- TASK-39152 re-pinned 74 of its 86 stale nodes; 12 product-defect nodes stay red and could not be laned yet anyway.

**AC#3, the push-to-dev job** (`library-shell-dev`, "Library shell (push to dev)", in `.github/workflows/derived-artifacts.yml`):
- `if: github.event_name == 'push' && github.ref == 'refs/heads/dev'`, `needs: [preflight]`. It is not in the required aggregate's `needs` and no verdict step reads it, so a red shard never blocks a PR. The workflow's existing push concurrency rule cancels a run when a newer push to dev arrives, as it does every push run.
- **Split.** A 14-shard matrix, `max-parallel: 7`, `fail-fast: false`, 20-min timeout. Each shard runs the UI lane's install line, collects the file (`pytest --collect-only -q`), and runs every 14th node, round-robin in collection order (`scripts/library_shell_push_shard.py`), serially with the UI lane's flags (`--timeout=180 --durations=0 --durations-min=0 --tb=short`).
  - Round-robin spreads one slow test's parametrized cases across shards. The `fifty_*` and breakpoint families run 80-190 s each under load.
  - Why 14: replayed from the three whole-file runs' per-node times (each node at its slowest, capped at 180 s), 14 shards carry at most 11.3 local min against 13.2 for 12 shards. At CI's measured upper bound of 1.0x local, plus about 1.7 min of setup, that is about 14 min a job, 6 min under the cap.
  - Why 7 at once: a dev push borrows half the runners a PR's UI lane takes (18) and finishes in two waves.
- **Deselected** by exact node id, in `scripts/library_shell_push_deselect.txt`, each under a one-line reason: the 12 product-defect nodes from TASK-39152's notes (TASK-34000.60's node among them). The shard script fails the job if a listed id is no longer collected, so the list cannot go stale silently. The coordinator replaces the reasons with follow-up task ids.
- **Per-shard times.** All 14 shards were run locally with the step's own commands (run.sh, 7 shards at once, each a serial pytest process, load 33-49).
  - Pytest minutes per shard, 1-14: 5.6, 5.1, 5.4, 6.6, 7.0, 7.6, 6.9, 8.0, 6.2, 6.0, 7.2, 6.1, 7.0, 6.1. Wall time with collection: 5.7-8.3 min.
  - Shards 1-10 hold 60 nodes each and shards 11-14 hold 59: 836 nodes, the file's 848 less the 12 deselected.
  - 830 passed and 6 failed. The 6 are watch items, below.
- **Runner-minutes per push** (estimate): the shards' local wall sum, 97.3 min, times TASK-39144's CI/local ratio of 0.51-0.99, plus 14 x 1.7 min of setup: about 73-120 runner-minutes per push, about 95 at the median ratio. The slowest job is estimated at 6-10 min.
- **Watch items.** These are load-intermittent nodes from TASK-39141's list. They stay in: CI runs each shard serially on its own runner.
  - Failed in the local 14-shard run: `test_library_shell_blank_note_escape_key_returns_to_list_without_crash`, `test_library_note_deliberate_scroll_override_replaces_responsive_memory`, `test_library_onboarding_unmount_flushes_latest_lifecycle`, `test_library_shell_note_explicit_save_calls_seam_and_bumps_version`, `test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata`, `test_library_note_same_side_resize_does_no_presentation_work[initial_size1-same_side_sizes1-False]`.
  - Failed in TASK-39152's whole-file runs: `test_library_note_breakpoint_round_trips_restore_every_region_focus_role`, `test_library_shell_blank_title_save_round_trip_agrees_with_the_row`, `test_library_conversation_retry_first_failure_has_no_applied_metadata`, and the 1-of-3 nodes listed there.
  - Three also failed alone, so they may turn the job red in CI: `blank_note_escape_key` (4 of 4 alone), `note_breakpoint_round_trips` (3 of 4) and `deliberate_scroll_override` (1 of 2).
- **CI contract.** `Tests/CI/test_derived_artifacts_workflow.py`:
  - pins the job list, with `library-shell-dev` last;
  - adds `test_library_shell_runs_by_node_on_dev_pushes_only_and_is_never_required` (the push-only `if`, out of the aggregate, the matrix and parallelism, the install line, the UI lane's flags, no xdist);
  - adds `test_library_shell_shards_partition_the_kept_nodes` (every kept node in exactly one shard, in order; a stale deselect entry or an out-of-range index refused);
  - adds `test_library_shell_deselect_entries_resolve_and_each_says_why` (each entry resolves through the census checker's `resolve_node` and has a reason line above it).
  - Each was shown red under a temporary edit, then restored: the job widened to PRs, a shard dropping a node, a reason line removed, and a stale id. The script itself also exits 1 on the stale id.

**AC#4.** None of the four files is in `pr-fast-lane` (checked over every step's `run` in `derived-artifacts.yml`: the privacy boundary is only in `architecture-guards`, the ingest flow only in `recovered-suites`, the rail minors file only in the UI census).

**Covering runs.** `Tests/CI` whole at `-n 4` (run.sh), with the AC#3 job and its three tests: 1243 passed, 1 skipped (`test_derived_artifacts_workflow.py` again after formatting: 70 passed). `./scripts/preflight.sh` through run.sh: all checks passed (census 1919, floor 1919).

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `scripts/library_shell_push_shard.py` (new), `scripts/library_shell_push_deselect.txt` (new), and `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py` (in TASK-39130's commit).
