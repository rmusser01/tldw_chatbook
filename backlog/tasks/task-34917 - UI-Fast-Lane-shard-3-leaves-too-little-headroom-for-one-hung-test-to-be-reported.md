---
id: TASK-34917
title: UI Fast Lane shard 3 leaves too little headroom for one hung test to be reported
status: Done
created_date: 2026-10-12 22:22
labels:
- ci
- tests
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-18 07:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI Fast Lane shard 3 has a 20-minute cap and ran with a median of 16.4 minutes and a maximum of 18.8 over 35 pull-request runs measured on 2026-10-12. The shard holds a Console test that hangs now and then (test_applying_on_anthropic_submits_none_of_the_hidden_fields). A hung private-profile test costs about 3 minutes 10 seconds before the wrapper reports it when the child stops at its own timeout, and about 3 minutes 18 seconds when it cannot exit. Added to the measured run times, that pushes the shard past its cap in a seventh to a fifth of runs (5 of 35 and 7 of 35 respectively, against 4 of 35 before the wrapper change in PR #3091), and the cancelled job then loses exactly the report that says where the test hung. Measured during the review of PR #3091 (TASK-34789).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shard 3's slowest run over at least 20 pull-request runs leaves room for one hung private-profile test to be reported before the cap
- [x] #2 The shards stay balanced, with no shard left closer to its cap than shard 3 is today
- [x] #3 The contract test that pins the shards reflects the new layout
- [x] #4 The 20-run measurement also records each job's queue delay (job start minus run creation) and the required check's wall clock (run creation to the aggregate job's completion), and that wall clock is not worse than the pre-change baseline in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure per-shard durations over the recent pull_request runs and per-entry durations from the shard logs (progress-line timestamps).
2. Replay the logs against 4/5/6 round-robin shards of the current census to find the smallest shard count whose heaviest shard leaves room for one hung private-profile test (about 3m18s) under the 20-min cap.
3. Change only the matrix (no census reorder, no new minutes) and record the replay in the workflow comment.
4. Update the contract tests / ADR-103 note; verify Tests/CI and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
UI Fast Lane matrix 4 -> 6 shards; census and its order unchanged; cap stays 20.

Measured (gh api jobs for 91 pull_request runs, 2026-10-12 19:39Z to 2026-10-14 14:26Z): shard medians 13.8 / 9.7 / 16.5 / 15.3 min, shard 3 p90 17.8 / max 19.7. Adding a hung private-profile test's ~3m10s-3m18s report to shard 3 crosses the cap in 33-36 of those 91 runs (the shard grew since the 2026-10-12 count of 5-7 of 35).

Replay: per-entry CI times read from 26 green runs' shard logs (104 logs; an entry's time = its last progress line's timestamp minus the previous entry's; consecutive node ids of one file share a line; the first entry's time sits in collection, ~46 s median), mapped against each run's head census (all 104 mapped). Re-dealing the current census (190 entries) and adding the measured ~1.0 min of job setup reproduces the measured 4-shard job times within 0.12 min median / 0.58 max (figures corrected in review of #3106, W6: the replay had looked up each run's own start-up time with an int key against string JSON keys, so every shard used the median start-up). Per shard median/max:
- 4: 14.0/16.3 | 10.0/11.2 | 17.3/19.7 | 15.7/17.7
- 5: 9.4/10.1 | 12.1/12.9 | 12.4/14.0 | 8.7/9.4 | 15.8/17.4 (the 5th shard collects test_css_generated_comments 2.9, test_console_rename_consistency 2.2, test_roleplay_hostile_text_surfaces 1.7, test_console_pending_interrupt_projection 1.7 min)
- 6: 8.6/9.4 | 8.6/9.4 | 12.5/14.1 | 8.5/9.3 | 12.1/13.1 | 10.0/11.0
Five does not leave room for the hung test (17.4 + 3.3 > 20); six does: the heaviest is shard 3 at 14.1, and it is still the shard holding test_console_settings_hidden_fields.py (index 62 of the census), so 14.1 + 3.3 = 17.4 leaves ~2.6 min. Seven is lighter at the top (13.3) but unbalanced (one shard at 4.1).

Expected after the change: shards median 8.5-12.5 min, max 14.1 (shard 3), every shard at least 5.6 min further from the cap than shard 3 is today (19.7 measured max). Cost: two more runners per PR (ADR-103 note); each shard pays ~1.8 min of fixed setup + collection, ~3.6 runner-min more per PR.

Risk not covered locally: every entry is re-dealt, so a file that leaned on a predecessor's leaked state can go red in the PR's CI (the lane comment says the split surfaces such bugs). The census could not be run locally under the scratch-profile pytest wrapper: Tests/UI fixtures fail at setup with RecoveryRequired('raw_source_selection_changed') (Tests/UI/conftest.py:193 importing app.py), so the PR's own CI run is the check of the new shards.

AC#1 and AC#2 are left unchecked: they ask for 20 measured runs of the new layout. Verify with the same gh api jobs query: max 'UI Fast Lane (3)' <= 16.7 min and no shard above shard 3's current max.

Contract: Tests/CI/test_derived_artifacts_workflow.py pins the matrix to [1..6] and the 20-min cap, the shard-coverage test now includes a 6-way split, and test_response_rules_critical_gate.py reads the shard count from the workflow instead of a literal 4. Tests/CI green (905 passed, 1 skipped).

Lesson added to backlog/docs/lessons-testing-evidence.md (round-robin N+1 is not monotonic; replay method).

Files: .github/workflows/derived-artifacts.yml, Tests/CI/test_derived_artifacts_workflow.py, Tests/CI/test_response_rules_critical_gate.py, backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md, backlog/docs/lessons-testing-evidence.md.
First run of the new layout (review of #3106, W3): run 38588725796 (PR #3106, head b1d4142d65, created 15:22:23Z 2026-10-14), every job green. Queue delay (job start minus run creation) / duration / done, in min: PR Fast Lane +2.4 / 16.6 / +19.0; Dreams tracking +3.3 / 6.8 / +10.1; UI shards 1-6 +10.1 / 6.3, +5.5 / 8.9, +5.2 / 10.0, +10.1 / 9.5, +6.2 / 11.8, +3.7 / 10.7 (shards done +14.4 to +19.6); Console P0 gate +5.1 / 16.5 / +21.6; PR Fast Lane (admission-sensitive) +9.0 / 14.9 / +23.9; aggregate started +26.1, done +30.9 (the required check's wall clock). The fan-out moved cost into queueing (2.4-10.1 min starts on a 40-job account pool shared with tldw_server), but the wall clock still improved on this run. Pre-change baseline, same definitions, over the 81 pull_request runs to 2026-10-14 with every lane and the aggregate finished: required check wall clock median 39.0 / p90 66.8 / max 256.1 min; last lane done median 33.9 / p90 61.4; per-job queue delay median 2.4 / p90 17.8 min (491 jobs). AC#4 compares the 20-run median and p90 against these. No max-parallel or concurrency change in this PR.
Interim measurement, 2026-10-14 19:48Z (TASK-34986 session): 12 completed pull_request runs of derived-artifacts.yml were created after #3106 merged (17:23:53Z), fewer than the 20 the ACs ask, so no AC is ticked. 11 of them finished every lane (one, 38606466442, was cancelled at 12.8 min). Queue delay of the lane jobs: median 2.0 / p90 21.9 / max 24.8 min after the run started, 3.8 / 22.0 / 40.2 after the run was created (one run, 38616567455, sat 35.9 min in a merge-queue approval hold). Required check wall clock (run created -> aggregate done): median 28.9 / p90 43.6 / max 57.7 min (11 runs), against the pre-change baseline of 39.0 / 66.8 / 256.1 (81 runs): better so far. UI shards (n=11-12) median / max: 7.4 / 11.1, 8.7 / 9.3, 10.8 / 11.9 (shard 3: <= 16.7 so far), 8.8 / 10.3, 11.4 / 16.8, 9.9 / 18.9. No shard above shard 3's old max of 19.7 so far, but shard 6 reached 18.9 on run 38613866408 by absorbing two over-limit tests, not one (corrected in review of #3117, item 3): the 180 s pytest-timeout hang in test_library_note_autosave_recovers, and test_session_close_pending_race_and_fleet_journeys at ~195 s (its private-profile child's own timeout fired and wrote its stacks, app code swallowed the exception, the child test passed with a 183.21 s call, the parent's 195 s wait then ran out while the child was exiting, and the ask's default action ended it, reported as 'still running after 195s; ended by the ask for its stacks before it could answer'). Both were reported before the cap, more than the one hung test this task made room for; shard 5's 16.8 (run 38616104543) was a slow runner (606 passed in 15m23s).
20-run measurement, 2026-10-15 (review of #3117, item 2; supersedes the interim above). Data: 57 completed pull_request runs of derived-artifacts.yml created after #3106 merged (2026-10-14 17:23:53Z); 51 on the #3106 layout, 49 of them with six UI shards (two runs of a ten-shard branch excluded), 46 with every lane and the aggregate finished. Baseline: the 81 pre-#3106 runs (2026-10-12 19:39Z to 2026-10-14 14:26Z) with every lane and the aggregate finished. Percentiles: statistics.quantiles(n=10, method='inclusive'); figures checked against the reviewer's and identical. AC#1 met: shard 3 n=47, median 10.8 / max 12.6 min (run 38687245721), so 12.6 + ~3.3 for a hung test's report = ~15.9, about 4 min under the 20-min cap. AC#2 met: shards median / max 7.4 / 11.1, 8.8 / 9.8, 10.8 / 12.6, 9.0 / 10.3, 11.8 / 16.8, 9.9 / 18.9; no shard above shard 3's old max of 19.7. The 18.9 is shard 6 on run 38613866408, which absorbed two over-limit tests (see the corrected interim note); without such runs the highest is shard 5's 16.8 (run 38616104543, a slow runner, 606 passed). AC#4 is NOT met: the required check's wall clock as recorded (run created -> aggregate done) went from median 39.0 / p90 66.8 / max 256.1 to 41.8 / p90 95.2 / max 290.7 min (46 runs). Without the approval hold (first lane job created -> aggregate done) it went from 36.4 / 40.1 / 50.7 to 26.0 / 57.2 / 71.2: the median improved, the tail got worse. Lane queue delay (job started - job created) went from median 0.1 / p90 6.5 / max 17.4 to 1.3 / 21.3 / 39.8 min (491 -> 461 jobs). The periods differ in load (more PR traffic, the merge queue on), so the comparison is confounded and does not isolate the fan-out; whether the tail is acceptable for the run-time headroom, or needs max-parallel / fewer runners, needs an owner decision. Task stays In Progress for that.
AC#4 re-measured on the layout dev runs now, 2026-10-18. Dev has run ten UI shards since #3057 (TASK-33007, commit e580d60155), so the six-shard figures above no longer describe dev. Data: every completed attempt of the pull_request runs of derived-artifacts.yml created from 2026-10-15 16:27Z (#3117 merged) to 2026-10-18 06:33Z: 90 runs, 124 attempts, read with gh api .../runs/<id>/attempts/<n>/jobs and the failed jobs' logs. First attempts on the ten-shard layout with every lane job and the aggregate finished as success or failure: 41 runs, 2026-10-15 17:56Z to 2026-10-18 06:02Z. Same definitions and percentiles as above (statistics.quantiles(n=10, method='inclusive')). Required check wall clock as recorded (run created -> aggregate done): median 25.2 / p90 40.5 / max 47.3 min, against the pre-#3106 baseline 39.0 / 66.8 / 256.1. Without the approval hold (first lane job created -> aggregate done): 24.9 / 38.6 / 42.4, against 36.4 / 40.1 / 50.7. AC#4 asks for queue delay as job start minus run creation: on these 41 runs, lane jobs started median 1.3 / p90 13.6 / max 27.8 min after their run was created (615 jobs), against median 2.4 / p90 17.8 before #3106 (491 jobs, from the first-run note above); the job-created figures above leave out the time before GitHub created the job. Lane queue delay from job creation (job started - job created): median 0.1 / p90 11.1 / max 22.9 min over 615 jobs, against 0.1 / 6.5 / 17.4 before #3106: measured that way queueing is still worse at the tail, but the wall clock is better than the baseline at the median, p90 and max, which is what AC#4 asks. The 2026-10-15 tail (p90 57.2) does not recur in these 41 runs; load differs between the periods, so this does not isolate the fan-out either. AC#1 and AC#2 still hold on ten shards: the hidden-fields test (census index 62) is still in shard 3, whose max is 15.7 min (n=69, median 14.1), within the 16.7 bar above; the heaviest shard is 9 at max 17.4 (medians 11.2-15.0 across shards), below shard 3's pre-change 19.7. The shards are heavier than the six-shard expectation because #3057 added 92 census entries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Re-dealt the UI Fast Lane census over more shards (six in PR #3106; dev has since gone to ten). The shard holding the hanging hidden-fields test now peaks at 15.7 min, leaving room for one hung test's report before the 20-min cap, and the required check's wall clock is better than before at the median, p90 and max; jobs still wait longer for a runner at the tail once GitHub has created them.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
