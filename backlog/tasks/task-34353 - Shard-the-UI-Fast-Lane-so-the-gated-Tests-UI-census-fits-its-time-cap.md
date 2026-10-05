---
id: TASK-34353
title: Shard the UI Fast Lane so the gated Tests/UI census fits its time cap
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-04 02:21'
updated_date: '2026-10-04 21:42'
labels:
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The UI Fast Lane runs the whole PR-gate census serially under a 20-minute cap. It already measured 14-19 minutes, and as the census grows PRs time out with zero failing tests (#2992 three times; #2953, #2977 and #2993 also hit 20 minutes), so the required check goes red on wall-clock alone. Splitting the census into shards that run in parallel keeps every file gated and census order within each shard, and restores headroom under the cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI lane runs the census as parallel shards; every census file runs in exactly one shard, in census order
- [x] #2 A red or timed-out shard still turns the required 'Derived artifacts reproduce from their sources' check red
- [ ] #3 Each shard finishes well inside the 20-minute cap on a PR carrying the Phase 6 census additions
- [x] #4 Tests/CI pins the shard shape (checker-picked shards, fail-fast off, minimal deps, no xdist) and runs the real shard command, instead of forbidding a strategy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split ui-fast-lane into a 2-shard matrix (fail-fast off); the shards come from strategy.job-index/job-total.
2. Keep the 20-min cap, minimal deps, no xdist; derived-artifacts' needs.ui-fast-lane.result already aggregates the matrix.
3. Replace the Tests/CI 'no strategy' pins with shard-shape pins; negative control against dev's workflow.
4. Measure shard times on this PR's CI; #2992 (Phase 6 census +5 files) rebases on it and verifies AC#3.

PR3011 follow-up plan (2026-10-04), before edits:
ADR required: no
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md (existing)
Reason: Tune the existing generic round-robin matrix under its unchanged census, 20-minute timeout, minimal dependency and fail-closed aggregation contracts. No context, trigger, prerequisite or dependency-boundary change.
1. Retain the current-head timeout annotation and logs: all1070 shard2 tests passed in1132.73s, but the20-minute job cap canceled pytest shutdown.
2. Reuse PR2918's exact one-line third-shard change, already prescribed by this task; retain all census files, order and original test assertions.
3. Run the two targeted CI contract files, workflow parsing, whitespace and Backlog guards; review independently. Do not rerun product or human/native/voice UAT.
4. Publish normally through existing PR3011 and require successful current-head hosted shards and aggregate before merge. Retain earlier completed tests and canceled-job attribution separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **The lane is a 2-shard matrix with fail-fast off.** `check_ui_pr_gate_census.py --shard INDEX TOTAL` prints a shard's files. The workflow writes that list to a file, so a failing checker fails the step under `bash -e`. It then refuses an empty shard, because bare `pytest` would collect the whole tree.
- **Deviation: shards are round-robin, not contiguous.** This PR's first CI run split the census contiguously, and the shards measured 531 s against 132 s of tests. The slow files sit together: the Console cluster at census positions 33-55, including session_tab_close at 139 s and Phase 6's five settings files at about 320 s.
  - Simulated on the Phase 6 census: contiguous gives 14.2 / 2.3 min, round-robin gives 6.8 / 9.7 min. With 3 shards, round-robin gives 4.7 / 6.2 / 5.7 min.
  - Each shard is still a subsequence of the census, in census order.
- **The required check still fails closed.** `needs.ui-fast-lane.result` is success only when every shard is.
- **Tests/CI changes:**
  - `test_ui_fast_lane_runs_the_census_in_serial_round_robin_shards` pins the shard shape.
  - `test_ui_gate_shards_cover_the_census_once_each_in_census_order` runs the real `--shard` command for 1, 2, 3 and 5 shards. Its negative control drops one file per shard and fails 4/4.
  - `test_ui_gate_shard_refuses_an_index_outside_the_split` covers an index outside the split.
  - `test_ci_queue_pressure_contract` accepts the matrix.
- **When a shard nears the cap, add a shard.** Do not raise the cap.

PR3011 capacity follow-up: GitHub run37232967293 at ae297322cd7be531b70192d62e910a4cc068c521 passed all1070 shard2 cases in1132.73s but canceled pytest shutdown at the20-minute job cap (1215s; explicit maximum-execution annotation). PR Fast Lane passed1249+135 and shard1passed465. Reused PR2918 exact one-line third-shard patch; complete census, ordering,20-minute cap,minimal dependencies,fail-fast:false and aggregate failure enforcement are unchanged. Both targeted CI contract files pass34tests in0.961s, including real1/2/3/5-shard coverage checks. Independent review found no actionable findings. Buddy product and test files are unchanged; no human/native/voice UAT or provider/audio call. Existing ADR103 applies; no new boundary, Python code or Bandit-applicable change. Source-bound receipt: Docs/Reviews/artifacts/buddy-uat-20261003/pr3011-ui-shard-budget-20261004.json. Hosted qualification on the new publication is pending; retain AC3 unchecked and In Progress until the required hosted duration outcome is observed.
<!-- SECTION:NOTES:END -->
