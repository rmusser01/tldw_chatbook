# CI lane headroom, 2026-11-28: the lanes near their time caps, and what each way out costs

Measured against `origin/dev` 2df861ec7d (the tip when the data was read). Job records of the 258 `pull_request` runs of `derived-artifacts.yml` created from 2026-11-24 to 2026-11-28 17:42Z, read with `gh api` (every attempt, each job execution counted once), plus the other pull-request workflows' jobs. The window starts 2026-11-26 00:00Z. 949 job logs and 87 job annotations were read. **Nothing here is applied.** A cap raise and an added runner are the owner's decision (ADR-103); every option below says which it needs.

PR 3804 (TASK-38716, TASK-42413) measured the PR Fast Lane and the recovered suites and now carries the owner's picks for them (a job of its own for the MCP workbench pair, the recovered suites at 40 minutes); PR 3803 (TASK-42719) did the same for the Dreams riders job (30 minutes). This note covers every other lane within 8 minutes of its cap, which is the margin TASK-35575 set for the admission-sensitive jobs (a hung test's report needs about 5.3 minutes at `--timeout=300`).

## What the owner has to decide

| # | Lane | Where it stands | Recommended | Needs |
| --- | --- | --- | --- | --- |
| 1 | `PR Fast Lane (admission-sensitive 4)`, TASK-42779 | One real cap hit (PR 3345's branch). On today's list the slowest of 99 runner draws needs 38.6 of 35 minutes (1 of the 99 modelled jobs over; about 3 in 100 by a fitted tail). Median 27.7, was 24.9 until 2026-11-27 | Move `Tests/Chat/test_workflow_planning_native.py` (315 s, 20% of the lane) into Console P0 `admission-sensitive 2`: slowest 31.2, 3.8 under, median 22.5 | No runner, no limit. PR 3345 already makes this exact move |
| 2 | Console P0 `admission-sensitive`, TASK-42533 AC #1 | 72 jobs on the split list: median 24.4, slowest 29.7, 5.3 under. AC #1 asks 8. Its "34.2 max" was a job on the unsplit list | Move three wrapped-collaborator files (235 s) to another Console P0 profile: slowest 24.8, 10.2 under | No runner, no limit |
| 3 | Managed plugin qualification, macOS, TASK-41519 | Median 25.0, p90 28.7, max 29.6 of 30. 2 real cap hits of 57, both PR 3345. Not a required check | Cap 30 to 40 | A limit. No runner |
| 4 | `PR Fast Lane (admission-sensitive 3)`, `(2)`, `(1)` | Slowest 32.2, 30.3, 28.3 of 35 (2.8, 4.7, 6.7 under). No cap hit in any | Nothing now. The next file goes to a lane with room | None |
| 5 | Architecture-review tests, Architecture guard contracts, UI shards, eval recovery | Slowest 17.9, 15.6 of 20; shards 19.9 of 20 (2 real hits in 3,641 jobs); eval recovery 17.7 of 20 | Nothing | None |

Two things the owner should know before reading the tables. **The 8-minute margin is gone in every PR admission lane**, not only lane 4: it was set when lane 1 ran median 17.5 / max 20.0 (TASK-35575, 2026-10-19) and it now runs 25.8 / 28.4. At that margin the only change that restores lane 4 is a split (+1 runner); the move above restores the 4-minute floor of TASK-34986 and leaves the 8-minute margin to the owner's call. And **PR 3717 would undo option 1**: it appends a 16-test, 2.6-minute file (`Tests/UI/test_console_trace_recovery_after_provider_error.py`) to lane 4; on today's list that puts 13 to 22 of 82 jobs over the cap. It is 385 commits behind dev and conflicting; the file belongs in a lane with room.

## Every lane within 8 minutes of its cap

Finished jobs (success or failure) since 2026-11-26. "Cap hits" counts only cancellations whose annotation reads "exceeded the maximum execution time". "Slowest on today's list" is the slowest measured runner speed applied to the lane's current file list (method below); lanes without a modelled column show the observed maximum. "Under" is the cap minus that slowest.

| Lane | Cap | Finished | Median | p90 | Max seen | Cap hits | Slowest on today's list | Under |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| admission-sensitive 4 | 35 | 97 | 25.4 | 32.2 | 34.9 | 1 | 38.6 | -3.6 |
| UI Fast Lane (42 shards) | 20 | 3,639 | 12.3 | 14.8 | 19.9 | 2 (of 44 candidates) | shard 4: max 19.4 | 0.6 |
| Dreams riders (PR 3803) | 20 | 92 | 17.1 | 19.3 | 19.9 | 7 | | |
| PR Fast Lane (PR 3804) | 35 | 84 | 31.0 | 33.9 | 34.7 | 12 | | |
| Console P0 admission-sensitive | 35 | 99 | 23.5 | 26.7 | 34.2 (unsplit list) | 0 | 29.6 | 5.4 |
| Recovered suites (PR 3804) | 30 | 90 | 22.7 | 26.6 | 29.1 | 9 | | |
| admission-sensitive 3 | 35 | 97 | 26.0 | 29.4 | 32.4 | 0 | 32.2 | 2.8 |
| architecture-review tests | 20 | 99 | 13.4 | 15.1 | 17.1 | 0 | 17.9 | 2.1 |
| Architecture guard contracts | 20 | 99 | 12.4 | 14.1 | 15.6 | 0 | 15.6 | 4.4 |
| admission-sensitive 2 | 35 | 99 | 24.6 | 28.8 | 29.6 | 0 | 30.3 | 4.7 |
| admission-sensitive 1 | 35 | 100 | 26.0 | 27.6 | 28.4 | 0 | 28.3 | 6.7 |
| Dev smoke and affected contracts | 30 | 22 | 8.0 | 19.6 | 21.5 | 0 | | 8.5 |
| Dreams boundary | 35 | 96 | 22.3 | 23.9 | 26.1 | 0 | | 8.9 |

Lanes with room, for moves: Console P0 `admission-sensitive 2` (median 17.2, slowest 20.1, 14.9 under), Console P0 `restore` (median 17.1, max 19.5 over 32 jobs since 2026-11-27 15:00Z, 15.5 under; it ran about 3.3 before the traced actions file arrived), Console P0 `sandboxed` (7.0 / 9.1; sandboxed files only).

Other workflows (not in the required check; the only required context on `dev` is "Derived artifacts reproduce from their sources"):

| Job | Cap | Finished | Median | p90 | Max | Cap hits |
| --- | --- | --- | --- | --- | --- | --- |
| Managed plugin qualification, `qualify (macos-latest)` | 30 | 55 | 25.0 | 28.7 | 29.6 | 2 |
| Eval recovery (windows-2022) | 20 | 65 | 12.3 | 16.4 | 17.7 | 0 |
| Perf guard, UI latency guardrails | 20 | 125 | 5.7 | 12.8 | 14.6 | 0 |

`lint`, `merge-queue`, `tool-recovery`, `qualify (ubuntu)` and `qualify (windows)` end at least 15 minutes under their caps. `plugins-recovery-ui` (cap 25) peaks at 12.4.

## How the previous numbers misled, and what they are now

- **44 of 46 "cancelled at the cap" rows for the 20-minute lanes were one run cancelled under the owner's account** ("The run was canceled by @rmusser01"; run 46106568580, 04:57Z, 20.1 minutes in: 42 UI shards, architecture-review, Architecture guards). The 42 shard jobs never had a runner. Only shard 4 (`fu/r26-boot`, 2026-11-26, at 84%) and shard 34 (PR 3345, 2026-11-28, at 95%) hit the cap, both still printing when cut.
- **Of 11 cancellations at 5 minutes under the cap or later in the five lanes read in full, one was a cap hit** (lane 4, below). The other ten read "canceled by @github-actions[bot]" (a newer push) or "by @rmusser01".
- **Console P0 `admission-sensitive`'s 34.2** ran the unsplit list; the split (PR 3782) merged 2026-11-26 04:27Z.
- **Lane 4's list grew on 2026-11-27 17:26Z** (PR 3777 took `test_console_right_rail.py` from 37 to 70 tests, 176 s to 365 s). Median 24.9 over 59 jobs on the old list, 27.7 over 15 on the new.
- **The speed ratio uses only files whose test count matches the reference's.** With the right-rail file in (its content doubled), lane 4's slowest draw reads 34.9 instead of 38.6. See the lesson in `lessons-testing-evidence.md`.
- **PR 3804's lanes since its data (06:00Z)**: the PR Fast Lane has one more cap hit (12 of 96; run 46111377409, `codex/research-workspace-continuation`, cut at 95%, still printing), the recovered suites three more (9 of 99; PR 3345 at 99%, and two on `codex/research-workspace-continuation` and `ci/42719-dreams-riders-cap` that had printed `2272 passed` 2 to 3 seconds before the cut, so they needed about 30 minutes). All slow, none hung. PR 3803's own recovered-suites job is one of them.
- **No cap hit in any lane was a hang.** Each log was still printing when the cap cut it, or had finished its tests: lane 4's last progress line was 3.1 minutes before the cut (at 93%, inside a heavy file; the stall limit used is 5), both UI shards were printing at the cut (84% and 95%), the two macOS jobs had printed pytest's summary, and the PR Fast Lane and recovered-suites hits above likewise. The hang PR 3804 found (PR 3345's five PR Fast Lane jobs, silent for 11 to 19 minutes) has not appeared in these lanes.

## Lane 4 (TASK-42779)

On today's list (647 tests, 21 files, 1,583 s of tests plus 71 s of setup and collection; reference job 140732676095): the model reproduces the 15 jobs that ran this list (observed 27.7 / 33.1 / 33.5, model 27.6 / 33.0 / 33.4 for median / p90 / max). Runner speed over the 97 complete jobs: 0.76 to 1.40 times the reference (1.84 times); AWS 0.76 to 0.87, Azure 0.76 to 1.40, `westus` the slowest. The two PR 3345 jobs cut by the cap or by a newer push did not draw unusually slow runners: on the files they share with the reference they ran at 1.10 (the median job runs 1.01), and they would have needed 30.2 and 30.3 minutes on dev's list. They ran PR 3345's own 663-test list (its right-rail file took 482 s in both, against a 191 s median on other branches, its workflow-planning file 494 and 521 s against 303).

Options (82 jobs of branches other than PR 3345; adding its 17 jobs moves the median and p90 of the first four rows by at most 0.2 minutes and leaves the slowest and the one job over unchanged):

| Change | Runners | Median | p90 | Slowest | Under | Over the cap |
| --- | --- | --- | --- | --- | --- | --- |
| none (cap 35) | 0 | 27.7 | 32.4 | 38.6 | -3.6 | 1 of 82 |
| cap 35 to 40 | 0 | 27.7 | 32.4 | 38.6 | 1.4 | 0 |
| cap 35 to 44 | 0 | 27.7 | 32.4 | 38.6 | 5.4 | 0 |
| workflow-planning file (315 s) to Console P0 `admission-sensitive 2` | 0 | 22.5 | 26.2 | 31.2 | 3.8 | 0 |
| split after `test_console_runtime_lifetime.py` (713 s and 871 s) | +1 | 13.1 and 15.8 | 15.3 and 18.4 | 18.2 and 21.9 | 16.8 and 13.1 | 0 |
| PR 3717's file added (+156 s local / +230 s at CI speed) | 0 | 30.4 / 31.6 | 35.4 / 36.9 | 42.3 / 44.0 | -7.3 / -9.0 | 13 / 22 of 82 |
| the move, then PR 3717's file at 230 s | 0 | 26.3 | 30.7 | 36.6 | -1.6 | 1 of 82 |

- Cap 40 is the smallest change. It leaves 1.4 minutes on the slowest draw and cannot hold PR 3717's file. Cap 44 is the highest that stays below the 45-minute busy-job deadline of the self-hosted controller (ADR-359; `CI_TEST_RUNNER` is unset, so these run on hosted `ubuntu-latest`).
- The move costs no runner and no limit and fixes the cause: lane 4 had 9 targets on 2026-11-18 and has 21 now. Destination: `admission-sensitive 2` has 14.9 minutes under; with the file it ends at 26.3 (8.7 under, 91 jobs). PR 3345 carries the move with its own measurement ("8m42s in Admission 4; this profile took 13m34s on the same head"). It needs `Tests/CI`'s `ADMISSION_4_TARGETS`, the Console P0 pins (`test_console_p0_regression_gate.py`, `test_response_rules_critical_gate.py`) and a dated ADR-103 note. The right-rail file (345 s) would do as well as the planning file.
- The split is the only option that meets the 8-minute margin for lane 4. Cost: one more runner per PR (dev's tip counts 36 concurrent test runners, 37 with perf-guard; PR 3804's new job, the owner's pick of 2026-11-28, makes it 37 and 38; this one would make it 38 and 39, against the account ceiling of 40 in ADR-103), about 1.5 more runner-minutes a run, a verdict step in the aggregate and the `Tests/CI` job pins, and a full CI run to prove it.
- xdist was not measured: no Linux host here, and the lane notes pin a serial file order (TASK-32873's bootstrap-profile rule). TASK-42628's four workers are for `Tests/CI` only.

## Console P0 `admission-sensitive` (TASK-42533 AC #1)

Over 99 jobs since 11-26 (72 on the split list): median 23.4, p90 26.2, slowest modelled 29.6 (5.4 under); observed on the list 24.4 / 26.3 / 29.7. The profile's `test_console_failed_call_retry_ownership.py` (589 s, 210 tests) is 43% of it; TASK-42525 put it last in the head (round 26, 2026-11-24 to 25), expecting the head to reach 24 to 27 of 35. The profile it was split from, `admission-sensitive 2`, has room.

| Change | Runners | Median | p90 | Slowest | Under | Over the cap |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0 | 23.4 | 26.2 | 29.6 | 5.4 | 0 of 99 |
| cap 35 to 40 | 0 | 23.4 | 26.2 | 29.6 | 10.4 | 0 |
| three wrapped-collaborator files (235 s) to another profile | 0 | 19.6 | 22.0 | 24.8 | 10.2 | 0 |
| the retry file (589 s) in a job of its own | +1 | 14.0 and 10.8 | 15.6 and 12.0 | 17.6 and 13.5 | 17.4 and 21.5 | 0 |

Destinations, from the same method: `admission-sensitive 2` takes 235 s to 24.8 (10.2 under), 315 s (lane 4's file) to 26.3 (8.7 under), **both to 31.0 (4.0 under)**. It can hold one of the two moves at the 8-minute margin, not both. `restore` is a job-level estimate (about 24 with 235 s, 10 under): its four files are pinned as the restore selectors plus the traced actions file. The rule that a sandboxed target may not follow a bootstrap-profile one holds in every destination.

## Other lanes

- **admission-sensitive 3** (567 tests, 15 files): 69 jobs on today's list at 26.5 / 29.9 / 32.4. `UI/test_roleplay_item_list_app.py` is 578 s of 1,395 (41%). Cap 40 leaves 7.8 under; the file out of the lane, 15.5 under; a split (+1) 23.0 and 10.8. No cap hit has happened and a fitted tail puts it at 0.1%. Nothing needed now.
- **admission-sensitive 2** (1,563 tests, 49 files; collection alone takes 224 s) and **1** (775 tests, 30 files): slowest 30.3 and 28.3, no hit. Nothing needed now.
- **UI shards**: shard 4 runs median 16.1 / p90 18.3 / max 19.4, against 12.3 for the lane; 26 of 3,639 jobs ended at 18 minutes or more. PR 3345's branch is 15% of the jobs and 24 of the 60 that took 17 minutes or more. The owner-approved remedy is `scripts/ui_lane_shard_replay.py --rebalance` (TASK-39144), which needs no runner and no limit. Not run here.
- **architecture-review tests** is below the 4-minute floor on the model (2.1 under), with no cap hit in 99 jobs and a fitted tail near 0.1%; **Architecture guard contracts** is 4.4 under.
- **Managed plugin qualification on macOS (TASK-41519)**: the one step, a single pytest run of about 834 tests, took 17.4 to 28.3 minutes in the 7 logs read (the same tests, all on `westus`: 1.63 times). Both cap hits (2026-11-27 10:13Z and 20:15Z, PR 3345) had finished their tests: pytest's summary (824 passed and 1 failed in 28:09; 840 passed in 28:18) printed at the 30-minute mark and the cut came 1 to 2 seconds later. The job needs about 30 minutes there. A cap of 35 leaves 4.7 minutes on the slowest draw seen, 40 leaves 9.7. No runner, not required, and the repository is public.

## Who else edits these blocks (open pull requests, 2026-11-28)

| PR | State | Edit |
| --- | --- | --- |
| 3345 (`codex/shared-console-mobile-pr`, draft) | pushed today | Moves the workflow-planning file from lane 4 to `admission-sensitive 2`; +3 nodes in Architecture guards; two Round 27 trace files into Dreams boundary; 2 targets out of Dreams riders; `dev-light` cap 30 to 15 and `plugins-recovery-ui` 25 to 15 (old values kept in a conflict resolution, per PR 3804's note) |
| 3717 (`fix/task-33621.6-retry-capture-regression`) | ready, conflicting, 385 behind | +1 file in lane 4 |
| 3572, 3534, 3569 | ready, 2,178 to 2,253 behind | +2 targets in Console P0 (profile unmeasured), about 24 targets in the PR Fast Lane and `admission-sensitive 2`, UI shards 38 to 40 and +1 Architecture guards node |
| 3342, 3579 | draft | `runs-on` / `if` lines only |

Whichever lands second of PR 3345 and any workflow change for lanes 1 or 2 must be rebased by hand.

## Not done, found on the way

- PR 3345's lane 4 jobs ran 1.22 times as slowly as other branches' in the same Azure region (median of 16 comparisons), its right-rail file 385 s against 191 s and its workflow-planning file 407 s against 303 s; its other seven admission, Console P0 and architecture lanes ran at 0.93 to 1.07 times. Cause not diagnosed. It also hung five PR Fast Lane runs (PR 3804's note).
- `scripts/ui_lane_shard_replay.py --rebalance` for shard 4 (above).
- The lane table tool counts a duration window as a cap hit; it should read annotations. The tools and data are in the notes folder named below.

## Re-measuring

Stdlib-only scripts and the per-lane speed tables are in the session notes folder `side-session-notes/local_078e0d89-2920-4808-bef5-1af6c01e4c26/` (`ci-near-cap-tools/`, `ci-near-cap-data/`): `collect_lane_times.py` (job records), `fetch_lane_logs.py` (logs and annotations of finished and near-cap jobs), `lane_report.py` (speed population and option arithmetic against a complete log of today's list, same-test-count files only), `lane_overview.py`. Pick a median-speed complete log of the current list as the reference; check each cancellation's annotation first.
