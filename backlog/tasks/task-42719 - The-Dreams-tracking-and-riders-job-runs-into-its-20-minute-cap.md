---
id: TASK-42719
title: The Dreams tracking and riders job runs into its 20-minute cap
status: In Progress
assignee:
  - '@claude'
created_date: '2026-11-27 22:30'
updated_date: '2026-11-28 08:35'
labels:
  - ci
dependencies:
  - TASK-34929
references:
  - .github/workflows/derived-artifacts.yml
  - Tests/CI/test_ci_queue_pressure_contract.py
  - backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The "Dreams tracking tests and unlaned riders" job (`dreams-tracking` in `.github/workflows/derived-artifacts.yml`) is one of the jobs behind the required check on dev pull requests. It has a 20-minute cap. GitHub cancels a job that reaches its cap, and the required check then fails. Rounds 24 to 26 added riders to the job: it ran 735 tests on 2026-11-22 and runs 1,605 now. From 2026-11-25 to 2026-11-28 it finished 107 times at a median of 17.1 minutes and was cancelled at its cap 7 times, with no test failing. One of the seven was PR 3802, which changes only the CI tier policy and its tests. A cancelled job costs the pull request a rerun.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Where the job's time goes is measured from CI runs: per file, and between fast and slow runs of the same tests
- [x] #2 The owner has the options with the measured cost of each, and a recommendation
- [x] #3 The option the owner chooses is applied, and the contract tests in `Tests/CI` pin it
- [ ] #4 Over at least 20 full pull-request runs after the change, the job is never cancelled at its cap and its slowest run ends at least 4 minutes under it
- [ ] #5 No test leaves the required check
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check that nobody is on it: open pull requests and remote branches that edit the job, dev's history of the workflow, the board on every ref.
2. Read the job's times from the GitHub API for every pull-request run since 2026-11-20, and per-file times from job logs. Compare fast and slow runs of the same target list.
3. Cost each way out from those numbers: a higher cap, a split, a move of riders to a lane with room, xdist workers.
4. Put the options to the owner. A cap raise and an added runner have both been owner decisions before (ADR-103, 2026-11-19 and 2026-11-24).
5. Apply the choice with its `Tests/CI` pin and a dated ADR-103 note, run `Tests/CI` and preflight, then measure 20 runs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Steps 1 to 5 are done except the 20-run measurement (AC #4), which needs 20 full pull-request runs after this lands. The pull request first carried only these notes, so its CI was the light tier. The workflow change came in a second commit after the owner's answer, so from then on it runs the full CI.

**Decision.** The owner chose option 1 on 2026-11-28, answering "Raise to 30 min" to "Raise the riders job's cap from 20 to 30 minutes?". It is applied. Options 2 and 3 stay below as the alternatives if the job's cap is hit again.

**What changed.** One line in the workflow (`timeout-minutes: 20` to `30` in the `dreams-tracking` job, line 1733 on dev 2df861ec7d), the job's comment, the pin `assert job["timeout-minutes"] == 20` in `Tests/CI/test_ci_queue_pressure_contract.py`, and a dated ADR-103 note, as for the recovered suites (9b4c574416, 2026-11-19). With only the cap line changed, `Tests/CI` on Windows fails the 54 nodes that fail on untouched dev 2df861ec7d (53 in `test_tiered_ci.py`, which PR 3802 addresses, and 1 symlink-privilege node) plus exactly one more, `test_dreams_tracking_lane_is_one_serial_minimal_python_312_job`, the pin above. The job's `needs`, verdict step, runner count and ADR counts do not change. A change under `.github/` starts the full CI run on the pull request that carries it.

**Options.** Headroom is measured against the slowest run seen, which would have needed about 21 minutes. The house floor is 4 minutes under the cap (TASK-34986).

| | Change | Runners | Slowest job after | Under the cap |
| --- | --- | --- | --- | --- |
| 1 | Cap 20 to 30 minutes | none added | about 21 of 30 | 9 min |
| 2 | Split after the Workspaces files into two jobs | one added | about 12.0 and 10.5 of 20 | 8 min or more each |
| 3 | Move riders to Console P0 (sandboxed) | none added | about 13.7 of 20 (the Dreams file) or 15.5 of 20 (16 riders) | 6.3 or 4.5 min |
| 4 | xdist workers in this job | none added | not measured | not measured |

1. **A 30-minute cap.** The smallest change, and the one that cannot break a test. It adds no runner and no runner time. Costs: a job that really hangs holds its runner 10 minutes longer (a hung test is cut by its own `--timeout=180` first), and the room can be filled again. The recovered suites are the warning: the owner raised them from 20 to 30 minutes on 2026-11-19 with no runner added, and since 2026-11-26 they have finished 83 times at a median of 22.4 and a max of 29.1 minutes and been cancelled at 30 minutes 6 times. The job's comment already said to re-measure in CI before adding more, so with a 30-minute cap a new rider should still start from the job's recent times (`gh api repos/<owner>/<repo>/actions/runs/<id>/jobs`). The owner also raised the macOS plugin recovery job to 25 on 2026-11-26. A 25-minute cap would leave 4 minutes.
2. **A split.** Cutting the target list after `Tests/Workspaces/test_workspace_create_modal.py` gives the most even halves: 400 and 360 s of tests in the fastest log, 630 and 537 s at the slowest seen. Each half keeps its order, so the Tool_Packs and storage-lease rules still hold inside it. It adds one job: 58 eligible test jobs for a full run and 37 concurrent runners, 38 with perf-guard (ADR-103 counts 57, 36 and 37 today), and about 1.4 runner-minutes for the second setup and collection. It needs a new job, a verdict step in the aggregate, and pins in several `Tests/CI` files (job order, verdict table, runner routing, lane tables, the ADR's counts). The second half would run without the first for the first time, which only a full CI run proves. The owner chose this shape for Console P0 on 2026-11-24 ("Split, +1 runner"), and the 2026-10-14 rule in TASK-34986 is to add a shard rather than raise the minutes.
3. **A move into Console P0 (sandboxed).** That profile has the same flags and install line, and finished 90 times since 2026-11-26 at a median of 6.9 and a max of 8.5 of its 35 minutes. The Dreams tracking file alone is 272 to 433 s (253 tests), so the job would end near 13.7 minutes at worst and Console P0 (sandboxed) near 15.7 of 35. Moving 16 riders instead (the create dialog, fleet messages, the contention table and the 13 first-chat files, 208 to 322 s) keeps the job's name and leaves 4.5 minutes under the cap. No runner and no limit changes. Costs: pins change in the Console P0 lane table and the sole-owner checks, the job's name stops matching its contents if the Dreams file leaves, and the moved files share a process with that gate's files for the first time, which only a full CI run shows.
4. **xdist.** Not measured: there is no Linux host here, and a number needs a full CI run. The job's own comment says a Tool_Packs file that read config in-process "would pass at -n 4 locally (another worker) and fail only here, serially" (TASK-32873), so workers would hide a check the serial process makes. It would also change the pinned argv and the install line.

**Recommendation, taken: option 1.** It is the smallest change, it cannot break a test, and it adds no runner. This job never sets the required check's wall clock (below), so a longer cap costs nothing there. Option 2 is the choice if the 20-minute caps should stay.

**In flight: nothing.** None of the 40 open pull requests changes this job's cap or target list. Of the 252 remote refs, which include the heads of 39 of those pull requests, five edit the job's block against their merge base with dev, and each edits one line that is not its cap or targets: `runs-on` on `codex/fix-eval-recovery-ci` (PR 3342, a draft) and the `if` line on `codex/light-ci-release-qualification` (PR 3579, a draft) and three `codex/light-ci-*-20261118` diagnostic branches. PR 3345 edits other jobs of the workflow, and PR 3575's branch does not change the workflow file. `git log origin/dev -S"unlaned riders"` on the workflow finds only e0f517f393, which named the job. No task about this cap exists on dev or on any remote ref. The highest task id cited anywhere is 42689 (PR 3802), so 42719 is that plus the 30-id lead.

**How it was measured.** GitHub API job records for the 467 `pull_request` runs of `derived-artifacts.yml` created from 2026-11-20 to 2026-11-28 05:50 UTC, each job execution counted once (a rerun of failed jobs repeats the others' records). A cancellation counts as "at the cap" only when the job's annotation says "The job has exceeded the maximum execution time of 20m0s". One more job ended at 20.1 minutes because its run was cancelled by hand. Per-file times come from the progress-line timestamps in 14 job logs: a line is stamped when its last test ends, so a file's time is the gap to the line before it.

**The job's time.** Minutes from job start to job end, jobs that finished:

| UTC day | finished | min | median | max | cancelled at the cap |
| --- | --- | --- | --- | --- | --- |
| 2026-11-20 | 41 | 4.8 | 6.4 | 7.8 | 0 |
| 2026-11-21 | 39 | 5.0 | 6.5 | 7.6 | 0 |
| 2026-11-22 | 49 | 4.9 | 6.7 | 8.0 | 0 |
| 2026-11-23 | 63 | 6.4 | 8.4 | 10.4 | 0 |
| 2026-11-24 | 53 | 7.0 | 9.9 | 19.6 | 0 |
| 2026-11-25 | 24 | 12.5 | 16.2 | 19.6 | 1 |
| 2026-11-26 | 36 | 13.6 | 18.1 | 19.9 | 3 |
| 2026-11-27 | 44 | 13.1 | 17.2 | 19.7 | 0 |
| 2026-11-28, to 05:50 | 3 | 13.2 | 16.9 | 19.3 | 3 |

- Since 2026-11-25: 107 finished, median 17.1, p90 19.3, max 19.9 minutes. 41 of them took 18 minutes or more and 23 took 19 or more. 7 of 114 (6.1%) were cancelled at the cap.
- The 26 jobs since 14:00 UTC on 2026-11-27, on dev's latest list: 23 finished (median 16.5, p90 18.9, max 19.5 minutes) and 3 (11.5%) were cancelled at the cap.
- The seven: PR 3771 runs 45661114677 and 45807634551; PR 3575 runs 46074517470 and 46104041876; PR 3802 run 46105646288; runs 45733253721 and 45733269086 of two branches without a pull request. No test had failed in any. Four had printed pytest's passing summary (1,137 to 1,149 s) before the cancel, and in 46105646288 every step had finished at 19 min 59 s. Three were cut with 44 to 232 tests left; the two of those with 44 left were in their last file and needed about 21 minutes. 45733269086 ran a longer list (1,711 tests, with the failed-call retry file that TASK-42525 moved out of this job).
- PR 3575's job took 13.8 and 16.9 minutes on two runs on 2026-11-27 and was cancelled on the next two.

**Where the time goes.** Eleven logs from 2026-11-26 to 11-28, all with the job's current 42 targets (1,600 to 1,616 tests, the difference being test edits on the pull requests):

- Setup (checkout, Python, pip install): median 52 s, p90 58 s over the 107 jobs. pytest start-up and collection: 28 to 40 s.
- Tests: 761 s in the fastest log. The two slowest that finished took 1,104 and 1,107 s, and the slowest cut run projects to about 1,160 s.
- By file, in the fastest log and in the 1,104 s log: the Dreams tracking file 272 and 377 s (253 tests, about 35% of the test time), the workspace create dialog 70 and 105 s, durable fleet messages 42 and 70 s, the character cadence file 42 and 60 s, the read-then-write contention table 41 and 56 s, native-76 compatibility 36 and 52 s. The other 51 files take 258 and 384 s together.
- No single test is slow. The slowest took 6.4 s on a slow runner. Inside the largest file the cost is spread too: in a local run on a loaded Windows host (2,672 s, so not comparable in seconds) fixture setup was 18% of the file and its ten slowest tests 16%, with no one fixture or test above 3% (68 of 2,672 s at most).

**Why runs differ: the runner, not a few files.** In the 1,104 s log the whole step took 1.45 times the fastest log, and each of the 16 heaviest files took 1.33 to 1.67 times what it took there. Start-up and collection scaled the same way. The fastest job on the current list took 13.1 minutes and the slowest would have needed about 21, so the 20-minute cap admits a runner about 1.5 times slower than the fastest and no more. The time of day explains little: the step's median was 14.7 to 17.0 minutes in each four-hour window.

**Why it grew.** The job ran 735 tests on 2026-11-22 (its own comment) and runs 1,605 now. Each rider was budgeted in the job's comment from one local time and a ratio. The last note projected "about 11.3-15.9 of its 20 min" and asked for a CI re-measure before more was added. That re-measure is the table above.

**The job's length does not delay the required check.** Over 103 full-run attempts since 2026-11-26, the time from the first lane's start to the last lane's end was a median of 135 minutes (p90 208). A UI shard finished last in 94 of them and this job in none. Only its cancellation costs anything.

**Seen in the same data, not part of this task.** Since 2026-11-26, by annotation: the PR Fast Lane was cancelled at its 35-minute cap 11 times of 86 (finished: median 31.2, max 34.7 minutes), and the recovered suites at their 30-minute cap 6 times of 89. One UI shard job in about 3,200 was cancelled at its cap.

**Measuring AC #4 after this lands.** List the `pull_request` runs of `derived-artifacts.yml` (`gh api repos/<owner>/<repo>/actions/workflows/derived-artifacts.yml/runs?event=pull_request`), read each run's jobs (`runs/<id>/attempts/<n>/jobs`) and take this job's `started_at` and `completed_at`. Confirm a cancellation from the job's annotation (`check-runs/<job id>/annotations`): a run cancelled by hand also ends near a cap. The criterion holds when 20 full runs show no cancellation at the cap and the slowest took 26 minutes or less.

Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`, `backlog/docs/lessons-testing-evidence.md` and this task.

**2026-11-28, later: combined.** The owner asked for PRs 3803, 3804 and 3805 to be combined before merging ("Combine them"). PR 3804 now carries this task's two commits, so PR 3803 is superseded by it; AC #4's 20 full runs are counted from when PR 3804 lands.
<!-- SECTION:NOTES:END -->
