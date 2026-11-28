---
id: TASK-42779
title: "The admission-sensitive 4 PR lane can end over its 35-minute cap, and the PR admission lanes have lost their 8-minute margin"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-28 11:00
updated_date: 2026-11-28 11:00
labels:
- follow-up
- ci
- needs-owner-decision
priority: high
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
- backlog/docs/ci-lane-headroom-2026-11-28.md
- backlog/tasks/task-35575 - The-admission-sensitive-PR-job-keeps-room-for-a-hung-test-s-report-under-its-cap.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while measuring the lanes that run near their caps (2026-11-28, against dev 2df861ec7d; all lanes are in `backlog/docs/ci-lane-headroom-2026-11-28.md`). `pr-fast-lane-admission-4` ("PR Fast Lane (admission-sensitive 4)", `timeout-minutes: 35`) is a required check. On the target list it runs today, the slowest of the 99 hosted runners measured since 2026-11-26 needs 38.6 minutes, so a job drawing a slow runner ends at the cap with every test so far passing, and the required check goes red for a reason unrelated to the pull request. The lane has had one such cancellation (a PR 3345 job); a fitted tail puts the rate on today's list at about 3 jobs in 100.

The lane came into being on 2026-11-18 with 9 targets (16.7 minutes serial) as a "New job" the owner chose, and runs 21 targets now. Its median went from 24.9 to 27.7 minutes on 2026-11-27 when one merged pull request doubled a file in it (`Tests/UI/test_console_right_rail.py`, 37 to 70 tests). Lanes 1 to 3 (`pr-fast-lane-admission`, `-2`, `-3`) are 6.7, 4.7 and 2.8 minutes under their caps at the slowest runner. TASK-35575 (Done) made the admission-sensitive jobs keep their slowest run at least 8 minutes under the cap, which leaves room for one hung test's report at `--timeout=300`. All four have lost that margin since.

An open pull request would make lane 4 worse: PR 3717 appends a 16-test, 2.6-minute file to it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on the remedy for lane 4 (a move, a cap change or a split) is recorded in the task notes
- [ ] #2 Over at least 20 pull-request runs after the change, lane 4's slowest job ends at least 4 minutes under its cap (the floor of TASK-34986), or 8 if the owner keeps TASK-35575's margin for it; the window and the figures are recorded in the notes
- [ ] #3 Every target that ran in lane 4 before still runs in a required PR lane with the same flags; no test leaves the required check to achieve it
- [ ] #4 The job's timing comment in the workflow states the measured median and slowest on its current list and the window they came from, and the contract tests that pin the lane reflect the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check that nobody is on it: the board on every ref, open pull requests and remote refs that edit the job block, dev's history of its targets and cap.
2. Read every job of the four PR admission lanes since 2026-11-26 from the API and its log, with the check-run annotation of every cancellation near the cap. Put each job on its list (`collected N items`). Compare each log with a complete log of today's list over the files that have the same test count in both.
3. Cost each way out against the measured runner speeds on today's list: a higher cap, a move to a lane with room, a split, xdist.
4. Put the options to the owner. A cap raise and an added runner have both been the owner's decision before (ADR-103).
5. Apply the choice with its `Tests/CI` pins and a dated ADR-103 note, run `Tests/CI` and preflight, then measure 20 full pull-request runs (AC #2).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-28, measured. Plan steps 1 to 4 are done. Step 5 waits for the owner. The pull request that carries this note changes no workflow, so its own CI is the light tier.

**Awaiting the owner (AC #1).** Which option below. The recommendation is option 3, the one-file move. Raising a limit and adding a runner are both the owner's decision, so neither is applied; the move needs neither, but it changes the Console P0 profile map and ADR-103, so it is put to the owner too.

**Checked first.** No task on any ref (23,332 distinct task and draft blobs, 581 refs) is about lane 3's or lane 4's cap: TASK-34916, TASK-35575, TASK-34986 and TASK-40724 are Done, and TASK-38716's note only lists the lanes (PR 3804). TASK-42533 is the Console P0 profile's. Of 237 remote refs with the workflow file, three edit lane 4's block against their merge base with `origin/dev`: `codex/shared-console-mobile-pr` (PR 3345, a draft pushed today) takes `Tests/Chat/test_workflow_planning_native.py` out of it and into Console P0 `admission-sensitive 2`; `fix/task-33621.6-retry-capture-regression` (PR 3717, ready, conflicting, 385 commits behind dev, last pushed 2026-11-25) appends `Tests/UI/test_console_trace_recovery_after_provider_error.py` (16 tests, "about 2.6 min serially", per its own comment); and `codex/light-ci-managed-received-diagnostic-20261118` changes the `if` line. Lane 3's block is edited by seven refs, none of them the cap: `runs-on` or `if` lines, comments, or two nodes on `codex/native-phased-workflows-20261105` (no open PR). Dev's history shows the cap at 35 since the lane was made.

**Measured.** GitHub API job records of the 258 `pull_request` runs of `derived-artifacts.yml` created from 2026-11-24 to 2026-11-28 17:42Z, each job execution counted once; logs of every finished job of the four lanes (lane 4: 99, lane 3: 97, lane 2: 99, lane 1: 100) and every cancellation within 5 minutes of the cap. A cancellation is "at the cap" only when its annotation reads "The job has exceeded the maximum execution time of 35m0s". The one in lane 4 (job 140985539568, run 46146704663, PR 3345's head f47a4418ef, 35.2 minutes) is the only one in the four lanes; the other cancellations read "canceled by @github-actions[bot]" (a newer push), including a second lane 4 job of PR 3345 (140827843660, 33.6 minutes). Neither PR 3345 job drew an unusually slow runner: on the files they share with today's list they ran at 1.10 times the reference (the 97 complete jobs run 0.76 to 1.40, median 1.01), which would take 30.2 and 30.3 minutes on dev's list. They ran that branch's own 663-test list, on which its right-rail file took 482 s in both (191 s median on other branches) and its workflow-planning file 494 and 521 s (303 s).

Lane 4 on today's list (647 tests, 21 files, reference job 140732676095: 1,583 s of tests, 71 s of setup and collection). The model reproduces the 15 jobs that ran this list: observed median 27.7, p90 33.1, max 33.5, model 27.6, 33.0, 33.4. Runner speed over the 97 complete jobs runs 0.76 to 1.40 times the reference (1.84 times; AWS 0.76 to 0.87, Azure 0.76 to 1.40, `westus` slowest; pooled over eight lanes AWS ran 0.79 and Azure 1.01 of a lane's median, `westus` 1.09, `eastus` 1.10). On the list before PR 3777 (59 jobs): median 24.9, p90 28.9, max 34.9, the 34.9 on Azure `westus` at 1.40 times, which needs 38.6 on today's list. A lognormal fit to the 99 speeds puts the share over the cap at 3.2% (3.5% without PR 3345's jobs); the count over is 1 of 99.

| Lane (cap 35) | Targets | Finished | Median | p90 | Max seen | Slowest on today's list | Under | Cap hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| admission-sensitive 4 | 21 files, 647 tests | 97 | 25.4 | 32.2 | 34.9 | 38.6 | -3.6 | 1 |
| admission-sensitive 3 | 15 files, 567 tests | 97 | 26.0 | 29.4 | 32.4 | 32.2 | 2.8 | 0 |
| admission-sensitive 2 | 49 files, 1,563 tests | 99 | 24.6 | 28.8 | 29.6 | 30.3 | 4.7 | 0 |
| admission-sensitive 1 | 30 files, 775 tests | 100 | 26.0 | 27.6 | 28.4 | 28.3 | 6.7 | 0 |

**Where the time goes** (lane 4, reference log, 1,583 s): `UI/test_console_right_rail.py` 345 s (22%, 70 tests), `Chat/test_workflow_planning_native.py` 315 s (20%, 27 tests), `Chat/test_workflow_failed_phase_release.py` 248 s (16%), `UI/test_console_rail_reconciliation.py` 229 s (14%), `Chat/test_console_reference_custody.py` 86 s, `Chat/test_change_turn_tracking.py` 76 s, the other 15 files 284 s. The four heaviest are 72%. Their slowest-to-fastest ratio between the slowest and fastest quarters of runners is 1.4 to 1.5; no file is slow by itself.

**Options for lane 4** (82 jobs of branches other than PR 3345, speeds applied to today's list; "Under" is the slowest against the cap; the 4-minute floor is TASK-34986's, the 8-minute margin TASK-35575's):

| | Change | Runners | Median | p90 | Slowest | Under | Over the cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | none (cap 35) | 0 | 27.7 | 32.4 | 38.6 | -3.6 | 1 of 82 |
| 1 | cap 35 to 40 | 0 | 27.7 | 32.4 | 38.6 | 1.4 | 0 |
| 2 | cap 35 to 44 | 0 | 27.7 | 32.4 | 38.6 | 5.4 | 0 |
| 3 | `test_workflow_planning_native.py` (315 s) to Console P0 `admission-sensitive 2` | 0 | 22.5 | 26.2 | 31.2 | 3.8 | 0 |
| 4 | split after `test_console_runtime_lifetime.py`: job 1 (13 files, 713 s) and job 2 (8 files, 871 s) | +1 | 13.1 and 15.8 | 15.3 and 18.4 | 18.2 and 21.9 | 16.8 and 13.1 | 0 |
| 5 | PR 3717's file added to lane 4, at 156 s (its local time) and at 230 s (CI speed) | 0 | 30.4 and 31.6 | 35.4 and 36.9 | 42.3 and 44.0 | -7.3 and -9.0 | 13 and 22 of 82 |
| 6 | option 3, then PR 3717's file at 230 s | 0 | 26.3 | 30.7 | 36.6 | -1.6 | 1 of 82 |

1. **Cap 40.** The smallest change: no runner, no runner time, no test touched. It leaves the slowest draw 1.4 minutes, below the floor, and cannot carry PR 3717's file. A hung job holds its runner 5 minutes longer.
2. **Cap 44.** The highest that stays below the 45-minute busy-job deadline of the self-hosted controller (ADR-359; `CI_TEST_RUNNER` is unset, so the hosted `ubuntu-latest` runs these jobs, but ADR-103's 2026-11-13 note keeps long lanes under the deadline). 5.4 minutes under; below the 8-minute margin, and the lane has added 12 targets in 10 days.
3. **The move.** It takes out the cause (the lane has grown from 9 to 21 targets) and needs no runner and no limit. Console P0 `admission-sensitive 2` runs median 17.2, slowest 20.1 (14.9 under, 91 jobs); with the file it ends at median 22.3, p90 25.2, slowest 26.3 (8.7 under). PR 3345 makes this very move with its own measurement ("native workflow planning took about 8m42s in Admission 4 ... This profile took 13m34s on the same head"), so it can be taken out of that draft as a small pull request of its own. It needs `ADMISSION_4_TARGETS` and the Console P0 pins in `Tests/CI` (`test_console_p0_regression_gate.py`, `test_response_rules_critical_gate.py`, `test_ci_queue_pressure_contract.py`) and a dated ADR-103 note. Costs: the job's name stops matching all of its contents, the file shares a process with that profile's trace files for the first time, and it uses room that TASK-42533's AC #1 also wants (`backlog/docs/ci-lane-headroom-2026-11-28.md` has both). It leaves lane 4 3.8 minutes under: at the floor, not at 8. The right-rail file (345 s) would serve as well.
4. **The split.** The only option that meets the 8-minute margin for lane 4. One more runner per PR (dev's tip counts 36 concurrent test runners, 37 with perf-guard; PR 3804's new job, the owner's pick of 2026-11-28, makes it 37 and 38; this one would make it 38 and 39, against the account ceiling of 40 in ADR-103), about 1.5 more runner-minutes of setup per run, a verdict step in the aggregate and pins in several `Tests/CI` files (job order, verdict table, runner routing, lane tables, the ADR-103 counts), and a full CI run to prove it, because the second half runs without the first for the first time. The owner chose "New job" for this lane on 2026-11-18 and "Split, +1 runner" for Console P0 on 2026-11-24.
5. **PR 3717.** Its file, left in lane 4, puts 13 to 22 of 82 jobs over the cap whatever else is chosen (options 0 to 3). It should go to a lane with room (Console P0 `admission-sensitive 2` takes 156 to 230 s to 23.2 to 24.7, 10 to 12 under), or the author should be asked to.
6. **xdist** was not measured: there is no Linux host here, and the lane's notes pin a serial order (TASK-32873: sandboxed targets first, then the bootstrap-profile ones). TASK-42628's four workers are for `Tests/CI` only.

**Recommendation: option 3, and option 5's destination for PR 3717.** One file is a fifth of the lane, so moving it fixes the cause without a limit or a runner, and the owner chose cap raises (recovered suites, Dreams riders) and splits (Console P0) on other lanes for the reasons PR 3804 gives. If the owner wants lane 4 back at 8 minutes under, option 4 is the answer, and it should come with a dated note of the headroom (AC #4). If neither the move nor a runner is wanted, option 1 is the stop-gap. Lanes 1 to 3 need nothing now: no cap hit in any, fitted tails of 0.04% to 0.1%. The next file added to a PR admission lane should go to a lane with room, not to lanes 1 to 4.

**Not applied, and why.** Options 1, 2 and 4 each need the owner (a limit or a runner); option 3 changes the Console P0 profile map and an ADR, and a change under `.github/` starts the full CI run, so it waits for the choice. No patch is prepared. Whichever of PR 3345 and the change lands second must be rebased by hand.

**How it was measured.** As in TASK-38716's note: job records and annotations from the GitHub API; per-file times from the job logs (a progress line is stamped when its newline arrives, so a file's time is the gap to the line before it); runner speed from the files a log shares with the reference at 1 s or more and with the same test count in both, dropping files whose ratio is outside 0.4 to 2.5 times the log's median (a pull request changed them); each job's minutes on today's list are its own setup and collection plus its speed times the reference's file seconds. The same-test-count rule matters here: without it the right-rail file, doubled in size, makes the slowest draw read 34.9 instead of 38.6. The tools, the per-log speed tables and these tables are in the session notes folder (`ci-near-cap-tools/`, `ci-near-cap-data/`).
<!-- SECTION:NOTES:END -->
