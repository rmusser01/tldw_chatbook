---
id: TASK-39153
title: "Lane the four files TASK-39141 made runnable: test_library_shell split by node, test_library_ingest_flow, test_console_inspector_rail_minors and test_persistent_terminal_privacy_boundary"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-10 19:34'
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
- [ ] #3 `test_library_shell.py` runs on PR lanes, split by node so that `scripts/ui_lane_shard_replay.py` shows every UI shard's slowest replayed run still at least 5 min under the 20-min cap, with three timings of each changed shard recorded
- [x] #4 None of the four files runs on the fast PR contract
<!-- AC:END -->

## Implementation Plan

1. AC#2: append `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` at the end of the Architecture guard step (its bootstrap node after every sandboxed suite) and to its pin in `Tests/CI/test_ci_queue_pressure_contract.py`; time the step's own command serially 3 times.
2. AC#1: `test_library_ingest_flow.py` (Tests/integration, so not census-eligible) joins the recovered-suites step after its bootstrap_profile suites, guard-clean under `--storage-lease-guard`; `test_console_inspector_rail_minors.py` joins the UI census tail; time each 3 times.
3. AC#3: once TASK-39152's re-pins land, price `test_library_shell.py` by node from junit durations and compute the 36-shard floor before any re-deal (lessons-testing-evidence, TASK-36512): if the floor is over 13.3 pytest min, no order meets it and the shard count is an owner decision; otherwise append its green nodes at the census tail and run `scripts/ui_lane_shard_replay.py --rebalance 13.3` over recent 36-shard logs.
4. AC#4: none of the four files joins `pr-fast-lane`.

## Implementation Notes

**Status: three of the four files now run on PR lanes (AC#1, AC#2, AC#4). AC#3 is not met: no order of the 36-shard UI lane can hold `test_library_shell.py`. The shard count is an owner decision (NEEDS_CONTEXT).**

**AC#1.**
- `Tests/UI/test_console_inspector_rail_minors.py` is a whole-file entry at the UI census tail (bootstrap_profile, 5 tests; committed with TASK-39130's five nodes, floor 1913 -> 1919). Three serial runs with the lane's flags (run.sh, `--timeout=180`, load 72-88): 92.0 s, 72.7 s and 94.0 s of pytest, 5 passed each. `scripts/ui_lane_shard_replay.py --shards 36` over the 22 saved round-12 PR runs, every new entry charged at its slowest local time: max shard 12.3 -> 13.1 pytest min, under the header's 13.3.
- `Tests/integration/test_library_ingest_flow.py` is not census-eligible (the checker takes `Tests/UI/` only), so it joins the recovered-suites step (`PR lane (recovered suites)`), last, after its bootstrap_profile suites, under `--storage-lease-guard`; its pin in `Tests/CI/test_ci_queue_pressure_contract.py` follows. The file alone, serial, guard on: 13 passed in 55 s (load 31-34). The whole step's own command, serial, three times: 839 s, 915 s and 668 s of pytest (21 targets, 746 passed and 1 skipped each, load 60 -> 35). That job took 6.1-6.5 min in CI before (runs 42841528703, 42841511400) against its 20-min cap.

**AC#2.** `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` runs last in the Architecture guard step (its one bootstrap_profile node after every sandboxed suite) and in `ARCHITECTURE_GUARD_TARGETS`. The file alone, three serial runs: 50.7 s, 56.2 s and 46.0 s, 8 passed each (load 33-38). The whole step's own command (36 targets), serial, three times: 1,324 s (load 84 -> 46), 1,045 s (load 46 -> 38) and 591 s (load 38 -> 28). The new file passed in all three. Runs 1 and 2 each had one unrelated failure, `test_profile_owned_path_inventory.py::test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners` hitting the 180 s timeout under load (180.0 s call). That test runs before the new file, and run 3 passed all 441. The job took 7.1-10.3 min in CI against its 20-min cap.

**AC#3, the floor (lessons-testing-evidence, TASK-36512: compute it before any deal).**
- The current census replays to 420.6 pytest min over 36 shards after this round's six entries (max 13.1), so at most 36 x 13.3 - 420.6 = 58.2 min of room is left, and only under perfect packing.
- `test_library_shell.py` (848 nodes): its junit durations summed to 8,683 s in the base run at `-n 4` (load 70-100). A serial run of 40 random green nodes took 1.066 times their `-n 4` time (load 64-96), so the file is about 154 min serial locally.
- TASK-39144's lesson measured CI at 0.51-0.99 of local serial time (median 0.73). That puts the file at 79-153 min in CI, median 113 min, or 13.9-15.9 min per shard over 36, against 13.3. No order meets it; the lane needs about 38-44 shards (40 at the median), or a job of its own. Both are owner decisions (the 36-shard count is the owner's 2026-11-07 decision), so no `test_library_shell.py` node is in the census.
- TASK-39152 re-pinned 74 of its 86 stale nodes; 12 product-defect nodes stay red and could not be laned yet anyway.

**AC#4.** None of the four files is in `pr-fast-lane` (checked over every step's `run` in `derived-artifacts.yml`: the privacy boundary is only in `architecture-guards`, the ingest flow only in `recovered-suites`, the rail minors file only in the UI census).

**Covering runs.** `Tests/CI/test_ci_queue_pressure_contract.py`, `test_derived_artifacts_workflow.py`, `test_ui_lane_shard_count_statements.py`: 207 passed. `./scripts/preflight.sh` through run.sh: all checks passed (census 1919, floor 1919).

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py` (in TASK-39130's commit).
