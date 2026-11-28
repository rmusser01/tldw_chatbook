---
id: TASK-42413
title: "The recovered-suites PR lane runs within about five minutes of its 30-minute cap and keeps growing"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-22 21:17
updated_date: 2026-11-28 10:00
labels:
- follow-up
- ci
- tests
- needs-owner-decision
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths start at the repository root.

**Why.** "PR lane (recovered suites)" (`derived-artifacts.yml:1696`) is one serial job under `--storage-lease-guard`, in the required aggregate's `needs`. Its cap went from 20 to 30 minutes on 2026-11-19 (owner decision, the note at `:1713-1716`) after 5 of 24 runs hit the old cap. Every round since has added targets to it, and its time keeps rising. If a run hits the cap, the required check goes red for a reason unrelated to the PR.

**Measured.** Method, so the numbers can be reproduced: the Derived Artifacts runs with event `pull_request` created from 2026-11-20 00:00Z to 2026-11-22 23:59Z, as listed by `gh run list --workflow derived-artifacts.yml --limit 400` at about 21:50Z on 2026-11-22 (140 runs); for each, every attempt's "PR lane (recovered suites)" job from the Actions jobs API (`filter=all`) whose conclusion is `success` or `failure` (122 jobs, no API errors); duration is the job's `completed_at - started_at`; days are the UTC date of `started_at`.

| Day (UTC, job start) | Jobs | Median (min) | Max (min) |
|---|---|---|---|
| 2026-11-20 | 41 | 19.9 | 25.7 |
| 2026-11-21 | 37 | 20.6 | 24.6 |
| 2026-11-22 | 44 | 21.3 | 25.1 |

Overall: 122 jobs, median 20.9, 90th percentile 23.4, max 25.7 (run 44727522321, a success). A review of this filing, bucketing by job start date from its own run listing, got 42/40/46 jobs and medians 19.9/20.3/21.3; the difference is which runs each listing caught, and the trend is the same. TASK-33620.15.61's port (round 21) adds about 22 s in CI, which puts the projected worst at about 25.5 minutes of 30. TASK-41516's two files are already in the job, so they are in these numbers.

The job's own structure suggests a split: one invocation runs the sandboxed suites first, then the `bootstrap_profile` ones, which must not be followed by a sandboxed suite that reads config in-process (TASK-32873). Each block could be its own job. Each new job is one more runner per PR (ADR-103's notes), so the split is an owner decision; moving targets to lanes with room is the alternative.

**Coordination.** Another session has open work on `.github/workflows/derived-artifacts.yml` and `Tests/CI/test_ci_queue_pressure_contract.py`: coordinate with tldw-chatbook-30's open PRs (#3656, #3683, #3688) and its coming TASK-42281 lanes PR before editing either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The recovered suites' projected worst case is at least 5 minutes under the cap of every job they run in, measured over at least 20 non-cancelled PR runs after the change (or projected from per-target serial times when fewer runs exist), with the numbers in the notes
- [ ] #2 Every target that ran in the job before the change still runs in a required PR lane, with the same flags (`--storage-lease-guard`, the sandboxed-then-bootstrap order) wherever it runs
- [ ] #3 `RECOVERED_SUITES_TARGETS` and the other lane contracts in `Tests/CI/test_ci_queue_pressure_contract.py` match the workflow, and `Tests/CI` passes
- [ ] #4 The workflow note for the recovered suites states its current headroom, so the next addition can be checked against it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check that nobody is on it: open pull requests and remote refs that edit the job, dev's history of its targets and cap, the board on every ref.
2. Read every recovered-suites job since 2026-11-26 from the API and its log, and per-file times. Sort each cap cancellation into slow or hung. Compare fast and slow logs of the same list.
3. Time the measured runner speeds against the list the job runs now, and cost each way out: a higher cap, a move of the newest targets to a lane with room, a split.
4. Put the options to the owner. A cap raise and an added runner have both been the owner's decision before (ADR-103, 2026-11-19 and 2026-11-24).
5. Apply the choice with its `Tests/CI` pins and a dated ADR-103 note, run `Tests/CI` and preflight, then measure 20 full pull-request runs (AC #1).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-28, re-measured. Plan steps 1 to 4 are done. Step 5 waits for the owner. The pull request that carries this note leaves the workflow unchanged, so its own CI is the light tier. The PR Fast Lane is TASK-38716's, which holds the method in full (how jobs were counted, how a cut log is sorted into slow or hung, how runner speed is read); this note gives the recovered suites' numbers.

**Awaiting the owner.** Which option below. The recommendation is option 2, a cap raise from 30 to 40 minutes. A cap raise and an added runner are both the owner's decision, so neither is applied. (AC #1 is the measured headroom after the change, so it stays open.)

**Checked first.** This task is the only one on any ref about this job's cap. No open pull request (41) changes the cap. Of 236 remote refs with the workflow file, six edit the job's block against their merge base with `origin/dev`: `codex/shared-console-mobile-pr` (PR 3345, a draft pushed today) takes the slot-drop and child-preset-fallback files out and puts them in `dreams-boundary`; `ci/task-33620.15.61-catalog-tests-in-pr-lanes` adds four catalog files dev already has; and PR 3579 and three `codex/light-ci-*-20261118` diagnostic branches change the `if` line and carry targets dev already has. Nothing but this note changes the cap.

**Measured.** GitHub API job records of the 467 `pull_request` runs of `derived-artifacts.yml` created from 2026-11-20 to 2026-11-28 05:50 UTC, each job execution counted once. A cancellation is "at the cap" only when the job's annotation says "The job has exceeded the maximum execution time of 30m0s": all six below were read. From 2026-11-26, 89 jobs reached a verdict or the cap: 83 finished (median 22.4, p90 25.5, max 29.1 minutes) and 6 were cancelled at 30 minutes. All six ran the Round 27 list (2,274 to 2,285 tests), none was stalled (each log was still printing when it was cut), and they were cut at 97 to 100% of their tests. The sixth (job 140551047795, run 46011097714) had printed pytest's passing summary, 2,272 passed, 5 seconds after the cap and was cancelled 2 seconds later.

| UTC day | finished | median | p90 | max |
| --- | --- | --- | --- | --- |
| 2026-11-20 | 41 | 19.9 | 22.5 | 25.7 |
| 2026-11-22 | 49 | 21.3 | 24.2 | 25.1 |
| 2026-11-24 | 52 | 22.2 | 24.4 | 25.8 |
| 2026-11-26 | 39 | 22.2 | 24.1 | 25.6 |
| 2026-11-27 | 38 | 22.5 | 27.1 | 29.1 |
| 2026-11-28, to 06:00 | 6 | 23.3 | 27.9 | 28.4 |

The six cancellations: runs 46011097714 (branch fu/r27-capture), 46052026681 (PR 3575), 46052364475 (PR 3345), 46055517368, 46068251108 and 46084712272 (PR 3771), all started between 2026-11-27 15:05 and 2026-11-28 01:51 UTC.

**Why it jumped on 2026-11-27.** Round 27 added twelve targets: the slot-drop file, the child-preset-fallback file, six Console files (request provenance census, continuation merge, named agent, redirect cut, route census capture, local citation boundary), two hook-engine nodes, the installed-restore validation file and the index plan-pin checker tests. The list went from 2,024 tests in 51 files to 2,274 in 62. In the reference log they take 383.9 s of the 1,715 s of tests, 22%: slot-drop 157.0, local citation boundary 114.4, route census capture 49.6, child-preset fallback 29.5, installed restore validation 14.7, and the other six 18.7 together. The job ran 71 targets on 2026-11-18, the day before the cap went from 20 to 30 minutes, and runs 98 now.

**Where the time goes** (reference log: job 140551047795, 2,273 tests, 62 files, 1,715 s of tests plus about 90 s of setup and collection):

| Folder | Seconds | Share |
| --- | --- | --- |
| `Tests/Chat` (27 files) | 934 | 54% |
| `Tests/Backup_Recovery` (8 files) | 320 | 19% |
| `Tests/Performance` (3 files) | 228 | 13% |
| the other 24 files | 233 | 14% |

No file is more than 12% of it: `Tests/Chat/test_trace_v2_collaboration.py` 207.8 s, the skill-trust stock-owned file 187.5 s (14 tests, a private process each), slot-drop 157.0 s, local citation boundary 114.4 s, skills recovery controls 95.6 s, home citation retirement 92.5 s, automatic Library preparation 88.5 s, config sync lifetime 76.8 s, Library ingest runner 67.2 s. So there is no single file to fix.

**Why runs differ: the hosted runner, with a spread of 1.75 times.** Over the 89 jobs (84 complete logs, one of them the cancelled job that had finished its tests, and 5 cut ones), each job's seconds over the files it shares with the reference, divided by the reference's, run from 0.64 to 1.13 (median 0.91). The 11 AWS-hosted jobs ran at 0.65 to 0.74 and the 78 Azure ones at 0.64 to 1.13: `eastus` and `westcentralus` slowest (medians 1.02), `centralus` fastest (0.82). On the same 2,024-test list, six complete logs took 857 to 1,445 s of tests, 1.69 times. No log was hung.

**What the cancellations imply for today's list.** Of the 89 jobs, 67 ran the old list (2,024 to 2,077 tests), 2 an intermediate one and 20 the new one (2,274 or 2,285), so the observed 6 of 89 understates the rate. Timing the 89 measured runner speeds against the list the job runs now gives median 27.4, p90 30.8 and a slowest of 33.7 minutes: 16 of 89 (18%) need 30 minutes or more. That is the rate to expect once every pull request carries Round 27's targets. Of the 20 jobs that ran the new list, 6 were cancelled (30%; a small sample, so it is consistent with the 18%).

**This job never sets the required check's wall clock.** In 335 full-run attempts since 2026-11-20 a UI shard finished last in 333 and this job in none, so a longer cap costs no wait. A cancellation costs the author a manual rerun of the same 27 minutes.

**Options.** "Under the cap" is the slowest of the 89 speeds on today's list against the cap; AC #1 wants 5 minutes, the house floor is 4 (TASK-34986).

| | Change | Runners | Median | p90 | Slowest | Under the cap | Over the cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | none (cap 30) | none | 27.4 | 30.8 | 33.7 | -3.7 | 16 of 89 |
| 1 | cap 30 to 35 | none | 27.4 | 30.8 | 33.7 | 1.3 | 0 |
| 2 | cap 30 to 40 | none | 27.4 | 30.8 | 33.7 | 6.3 | 0 |
| 3 | PR 3345's two files (slot-drop, child-preset fallback: 186.5 s) to `dreams-boundary` | none | 24.5 | 27.6 | 30.2 | -0.2 | 1 of 89 |
| 4 | all twelve Round 27 targets (383.9 s) to lanes with room | none | 21.5 | 24.2 | 26.5 | 3.5 | 0 |
| 5 | two jobs, cut after `test_console_chat_controller.py` (1,000 s and 715 s) | +1 | 16.5 and 12.2 | 18.5 and 13.7 | 20.2 and 14.9 | 9.8 and 15.1 | 0 |

1. **Cap 35** leaves 1.3 minutes: below the floor and the task's AC.
2. **Cap 40.** The smallest change: no runner, no runner time, and it cannot break a test. It stays below the 45-minute busy-job deadline of the self-hosted runner controller (ADR-359), which ADR-103's 2026-11-13 note keeps long lanes under. Costs: a hung job holds its runner 10 minutes longer, and the room can be filled again. The owner raised this job from 20 to 30 on 2026-11-19, when it ran 76 targets; it runs 98 now, and Round 27's twelve used a good part of the new room in one day. A 40-minute cap holds about 6 minutes more work at the slowest runner. The job's comment already says to re-measure in CI before adding more.
3. **PR 3345's two files to `dreams-boundary`.** Free, and it is already in that draft (comment: "Recovered Suites hit 30 minutes at 99%"). It takes the slowest from 33.7 to 30.2: still 0.2 over, so not enough alone. `dreams-boundary` would go from a max of 24.5 to about 28 of 35. With option 2 it leaves 9.8 minutes.
4. **All twelve Round 27 targets to lanes with room.** They are 7.0 minutes at the slowest runner, and almost all are `bootstrap_profile` files, so they need a bootstrap lane: `dreams-boundary` (max 24.5 of 35) would end 3.5 under, below the floor, and so would this job. No other lane has the room.
5. **A split.** Cutting after `Tests/Chat/test_console_chat_controller.py` gives the most even halves: 1,000 s and 715 s in the reference log, 20.2 and 14.9 minutes at the slowest runner. Each half keeps its order, so the sandboxed-before-bootstrap rule (TASK-32873) still holds inside it. Costs: one more runner per PR (37 concurrent test runners, 38 with perf-guard, against the account ceiling of 40 in ADR-103) and about 1.4 more runner-minutes of setup; a new job, a verdict step in the aggregate and pins in several `Tests/CI` files (job order, verdict table, runner routing, `RECOVERED_SUITES_TARGETS`, ADR-103's counts); and the second half runs without the first for the first time, which only a full CI run proves. The owner kept this job as one serial job on 2026-11-19 and chose "Split, +1 runner" for Console P0 on 2026-11-24.

**Recommendation: option 2.** It is the smallest change and meets AC #1's 5 minutes (6.3), it adds no runner, and the owner chose a cap raise for this job on 2026-11-19 (and, per PR 3803's description, for the Dreams riders job, TASK-42719, on 2026-11-28). Option 3 is free and worth taking with it. If the owner would rather stop raising this cap, option 5 is the structural answer, and it should come with a dated note of the headroom (AC #4).

**Not applied, and why.** Options 1, 2 and 5 each need the owner: a limit or a runner. Edits to this job's block in PR 3345 overlap any workflow change made for it, so whichever lands second must be rebased by hand.

**How it was measured.** As in TASK-38716's note: job records and annotations from the GitHub API; per-file times from the 89 job logs of this job; runner speed against the complete log of the current list. The tools, the per-log speed table and the tables above are in the session's notes folder (`ci-lane-cap-tools/`, `ci-lane-cap-data/`).
<!-- SECTION:NOTES:END -->
