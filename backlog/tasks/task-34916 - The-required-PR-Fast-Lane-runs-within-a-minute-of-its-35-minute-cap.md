---
id: TASK-34916
title: The required PR Fast Lane runs within a minute of its 35-minute cap
status: Done
created_date: 2026-10-12 22:22
labels:
- ci
- tests
priority: high
assignee:
- '@claude'
updated_date: 2026-10-18 07:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR Fast Lane is one of the jobs behind the required check, and it has a 35-minute cap. Over 34 pull-request runs measured on 2026-10-12 its median was 31.6 minutes and its slowest 34.0; one run on PR #3091 took 34 minutes 10 seconds. When a job reaches its cap GitHub cancels it, and a cancelled job leaves no test report, so the next few slow tests added to the lane will turn into failures with no explanation, which the merge queue then counts against the PR. The owner's rule for CI time is to add a shard rather than raise the minutes. Measured during the review of PR #3091 (TASK-34789).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PR Fast Lane's slowest run over at least 20 pull-request runs leaves at least 5 minutes under its cap
- [x] #2 No test is dropped from the required check to achieve it
- [x] #3 The contract test that pins the lanes and their caps reflects the new layout
- [x] #4 The 20-run measurement also records each job's queue delay (job start minus run creation) and the required check's wall clock (run creation to the aggregate job's completion), and that wall clock is not worse than the pre-change baseline in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure: per-job and per-step durations of the PR Fast Lane over the recent pull_request runs (gh api jobs).
2. The lane is two serial pytest steps of about 15 min each; move the admission-sensitive step into its own job (new runner, same install line, its own pytest process as before), so the PR Fast Lane keeps only the sandboxed step. Cap stays 35; the new job gets its own cap sized from the measured step.
3. Add the new job to the required aggregator's needs with its own verdict step.
4. Update the contract tests that pin the lanes (test_ci_queue_pressure_contract, test_derived_artifacts_workflow, test_console_p0_regression_gate, test_personal_context_regression_gate) and ADR-103.
5. Verify: Tests/CI, YAML parse, actionlint if present, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Split the PR Fast Lane's two serial pytest invocations into two parallel jobs instead of raising the cap.

Measured (gh api .../actions/runs/<id>/jobs, 90 pull_request runs of derived-artifacts.yml with both steps completed, 2026-10-12 19:39Z to 2026-10-14 14:26Z): the job took median 31.5 / p90 34.5 / max 39.5 min and 4 of the 90 were cancelled at the 35-min cap. The two steps are halves of it: 'Run fast PR contract' median 15.2 / max 19.9 min (Tests/UI/test_mcp_workbench.py alone is ~8.8 min of it), 'Run admission-sensitive suites' median 15.4 / max 18.7 (test_console_runtime_ownership 4.3, test_roleplay_header 3.3, test_roleplay_hostile_names 2.1, test_console_chat_create_integration 1.9 min); setup ~0.8 min.

Change: new job pr-fast-lane-admission ('PR Fast Lane (admission-sensitive)') holds the admission-sensitive step with its argv byte-for-byte unchanged (checked by parsing both workflow versions), the same install line and the same 35-min cap the owner set for these suites (TASK-33910.2). It was always its own pytest process, and each process makes its own mkdtemp bootstrap profile (Tests/conftest.py), so a fresh runner changes nothing it relied on. Dropped its 'if: always()' (it is the only test step in its job now). The required aggregate needs it and has a 'Require successful PR fast lane (admission-sensitive)' verdict in the same !cancelled() shape.

Expected durations (replay of the same 90 runs: job minus the moved step, and setup plus the moved step): PR Fast Lane median 16.1 / p90 17.9 / max 20.8 min (14.2 min under the cap); admission job median 16.3 / p90 17.5 / max 20.1 min (one hung test at --timeout=300, ~5.3 min with the private-profile wrapper's report, still ends ~10 min under). Cost: one more runner per PR (ADR-103 note).

AC#1 is left unchecked on purpose: it is about 20 measured runs of the new layout, which only the PR's and later PRs' CI can produce. Verify with: gh run list --workflow derived-artifacts.yml --event pull_request --limit 40 --json databaseId, then the max of 'PR Fast Lane' job durations from gh api repos/rmusser01/tldw_chatbook/actions/runs/<id>/jobs must be <= 30 min.

Tests: Tests/CI + Tests/Packaging/test_python_runtime_floor.py green (910 passed, 1 skipped); negative controls: dropping the new job from needs, deleting its verdict, or weakening its verdict to == 'failure' each fail _assert_required_aggregation.

Files: .github/workflows/derived-artifacts.yml, Tests/CI/test_ci_queue_pressure_contract.py, Tests/CI/test_derived_artifacts_workflow.py, Tests/CI/test_console_p0_regression_gate.py, Tests/CI/test_personal_context_regression_gate.py, backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
First run of the new layout (review of #3106, W3): run 38588725796 (PR #3106, head b1d4142d65, created 15:22:23Z 2026-10-14), every job green. Queue delay (job start minus run creation) / duration / done, in min: PR Fast Lane +2.4 / 16.6 / +19.0; Dreams tracking +3.3 / 6.8 / +10.1; UI shards 1-6 +10.1 / 6.3, +5.5 / 8.9, +5.2 / 10.0, +10.1 / 9.5, +6.2 / 11.8, +3.7 / 10.7 (shards done +14.4 to +19.6); Console P0 gate +5.1 / 16.5 / +21.6; PR Fast Lane (admission-sensitive) +9.0 / 14.9 / +23.9; aggregate started +26.1, done +30.9 (the required check's wall clock). The fan-out moved cost into queueing (2.4-10.1 min starts on a 40-job account pool shared with tldw_server), but the wall clock still improved on this run. Pre-change baseline, same definitions, over the 81 pull_request runs to 2026-10-14 with every lane and the aggregate finished: required check wall clock median 39.0 / p90 66.8 / max 256.1 min; last lane done median 33.9 / p90 61.4; per-job queue delay median 2.4 / p90 17.8 min (491 jobs). AC#4 compares the 20-run median and p90 against these. No max-parallel or concurrency change in this PR.

Also seen on that run, outside this task: the Console P0 regression gate took 16m30s of its 20-min cap (it measured median 13.0 / max 18.7 over 89 runs before this PR). Neither TASK-34916 nor TASK-34917 covers it; the coordinator is filing a task.
Interim measurement, 2026-10-14 19:48Z (TASK-34986 session): 12 completed pull_request runs of derived-artifacts.yml were created after #3106 merged (17:23:53Z), fewer than the 20 the ACs ask, so no AC is ticked. 11 of them finished every lane (one, 38606466442, was cancelled at 12.8 min). Queue delay of the lane jobs: median 2.0 / p90 21.9 / max 24.8 min after the run started, 3.8 / 22.0 / 40.2 after the run was created (one run, 38616567455, sat 35.9 min in a merge-queue approval hold). Required check wall clock (run created -> aggregate done): median 28.9 / p90 43.6 / max 57.7 min (11 runs), against the pre-change baseline of 39.0 / 66.8 / 256.1 (81 runs): better so far. PR Fast Lane: n=11, median 16.5 / p90 17.7 / max 22.5 min (<= 30 so far; the 22.5 is run 38611134368, 2169 passed in 21m15s, a slow runner, not a hang). PR Fast Lane (admission-sensitive): median 16.4 / max 18.0. Re-measure at 20 runs with the scratchpad's measure.py approach: gh run list --event pull_request + gh api .../runs/<id>/jobs, runs created after 17:23:53Z.
20-run measurement, 2026-10-15 (review of #3117, item 2; supersedes the interim above). Data: 57 completed pull_request runs of derived-artifacts.yml created after #3106 merged (2026-10-14 17:23:53Z); 51 on the #3106 layout, 49 of them with six UI shards (two runs of a ten-shard branch excluded), 46 with every lane and the aggregate finished. Baseline: the 81 pre-#3106 runs (2026-10-12 19:39Z to 2026-10-14 14:26Z) with every lane and the aggregate finished. Percentiles: statistics.quantiles(n=10, method='inclusive'); figures checked against the reviewer's and identical. AC#1 met: PR Fast Lane n=47, median 16.6 / max 22.5 min (run 38611134368, green: 2169 passed in 21m15s on a slow runner), 12.5 min under the 35-min cap (the AC asks 5); PR Fast Lane (admission-sensitive) median 16.8 / max 20.3 (run 38687245721). AC#4 is NOT met: the required check's wall clock as recorded (run created -> aggregate done) went from median 39.0 / p90 66.8 / max 256.1 to 41.8 / p90 95.2 / max 290.7 min (46 runs). Without the approval hold (first lane job created -> aggregate done) it went from 36.4 / 40.1 / 50.7 to 26.0 / 57.2 / 71.2: the median improved, the tail got worse. Lane queue delay (job started - job created) went from median 0.1 / p90 6.5 / max 17.4 to 1.3 / 21.3 / 39.8 min (491 -> 461 jobs). The periods differ in load (more PR traffic, the merge queue on), so the comparison is confounded and does not isolate the fan-out; whether the tail is acceptable for the run-time headroom, or needs max-parallel / fewer runners, needs an owner decision. Task stays In Progress for that.
AC#4 re-measured on the layout dev runs now, 2026-10-18. Dev has run ten UI shards since #3057 (TASK-33007, commit e580d60155), so the six-shard figures above no longer describe dev. Data: every completed attempt of the pull_request runs of derived-artifacts.yml created from 2026-10-15 16:27Z (#3117 merged) to 2026-10-18 06:33Z: 90 runs, 124 attempts, read with gh api .../runs/<id>/attempts/<n>/jobs and the failed jobs' logs. First attempts on the ten-shard layout with every lane job and the aggregate finished as success or failure: 41 runs, 2026-10-15 17:56Z to 2026-10-18 06:02Z. Same definitions and percentiles as above (statistics.quantiles(n=10, method='inclusive')). Required check wall clock as recorded (run created -> aggregate done): median 25.2 / p90 40.5 / max 47.3 min, against the pre-#3106 baseline 39.0 / 66.8 / 256.1. Without the approval hold (first lane job created -> aggregate done): 24.9 / 38.6 / 42.4, against 36.4 / 40.1 / 50.7. AC#4 asks for queue delay as job start minus run creation: on these 41 runs, lane jobs started median 1.3 / p90 13.6 / max 27.8 min after their run was created (615 jobs), against median 2.4 / p90 17.8 before #3106 (491 jobs, from the first-run note above); the job-created figures above leave out the time before GitHub created the job. Lane queue delay from job creation (job started - job created): median 0.1 / p90 11.1 / max 22.9 min over 615 jobs, against 0.1 / 6.5 / 17.4 before #3106: measured that way queueing is still worse at the tail, but the wall clock is better than the baseline at the median, p90 and max, which is what AC#4 asks. The 2026-10-15 tail (p90 57.2) does not recur in these 41 runs; load differs between the periods, so this does not isolate the fan-out either. AC#1 still holds on ten shards: 'PR Fast Lane' n=67, median 17.5 / max 20.1 min (cap 35); 'PR Fast Lane (admission-sensitive)' n=68, median 16.8 / max 19.2.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Split the PR Fast Lane's admission-sensitive suites into their own required job (PR #3106). Its slowest run is now 20.1 min against the 35-min cap, and the required check's wall clock on dev's current layout is better than before the split at the median, p90 and max; jobs still wait longer for a runner at the tail once GitHub has created them.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
