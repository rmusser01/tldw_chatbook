---
id: TASK-42533
title: "The admission-sensitive Console P0 job runs 27-34 minutes of its 35-minute cap"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-24 09:14
updated_date: 2026-11-28 12:00
labels:
- follow-up
- ci
- needs-owner-decision
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_response_rules_critical_gate.py
- Tests/CI/test_console_p0_regression_gate.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the coordinator's review of TASK-42472 (2026-11-24).

**Why.** `console-p0-gate (admission-sensitive)` took 27-34 min on recent PR runs against its 35-min `timeout-minutes`: 45298294580 (27 min), 45278583267 (32), 45300571348 (33), 45280698401 (34). A slow runner or a few more entries will cancel it at the cap. TASK-42472 added one file there (`Tests/Chat/test_console_trace_model_retries.py`, 4 tests, 10.2 s locally). Its other new lane file, `Tests/Chat/test_console_failed_call_retry_ownership.py`, went to the Dreams tracking riders instead. That file reads config: a first attempt to move it to the sandboxed step without its marker passed only because a swallowed `RecoveryRequired` fell back to the default.

**Options.** Move files whose `bootstrap_profile` marker is not needed to the sandboxed step (5-6 min in CI), or split the admission-sensitive profile in two. Each file in that step runs in the shared bootstrap profile for TASK-32873's reason. A moved file must therefore be shown to read no config in the per-test sandbox. A green run is not proof: record `get_cli_setting` calls and `RecoveryRequired` raises, and require zero raises (see the lesson "Green without `bootstrap_profile` does not prove a file reads no config").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Over the next 10 PR runs after the change, the admission-sensitive profile's slowest run ends at least 8 minutes under its cap
- [x] #2 Each moved file passes in its new invocation, and a recorder shows no swallowed `RecoveryRequired` there (any file still needing `bootstrap_profile` stays out of sandboxed invocations)
- [x] #3 `Tests/CI/test_response_rules_critical_gate.py` and `Tests/CI/test_console_p0_regression_gate.py` match the new lane lists
<!-- AC:END -->

## Implementation Plan

1. Measure: per-file times of the admission-sensitive step from four round-24 CI logs (45278583267, 45280698401, 45298294580, 45300571348), and the other legs' and jobs' times, to see where the 27-34 min goes and which jobs have headroom.
2. Choose on that evidence between a split, moving files, or speeding up a few slow files; never just raise the cap.
3. Split: a fourth `console-p0-gate` matrix profile, `admission-sensitive 2`, with its own bootstrap-profile pytest step. It takes the step's tail as one contiguous block (the round-16 trace files, the capture file, the Close-drain groups and the PR3023 witnesses, still last), so every target keeps its isolation profile and runs exactly once. Job-level `if`, `needs`, cap and the aggregate verdict are shared by all matrix legs, unchanged.
4. Contract tests: pin the new step and matrix in `Tests/CI/test_response_rules_critical_gate.py` (`SELECTION`, `INHERITED`, step count), `Tests/CI/test_console_p0_regression_gate.py` (lanes, bootstrap-OK set), `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_self_hosted_runner_routing.py` and `Tests/CI/test_derived_artifacts_workflow.py` (step list, PR3023 step); show removing or commenting out a new-step target turns the contract red. Update ADR-103's runner counts (pinned by `Tests/CI/test_ui_lane_shard_count_statements.py`) and the self-hosted README.
5. Time both legs locally (serial, the step's flags, via `r19-step.sh`) and on Linux (ubuntu:24.04, ext4 volumes, uid 1001, CI's install line, network off); project each leg's CI time from the per-file CI times.
6. Run `Tests/CI`, `Tests/Architecture` serially, preflight, ruff F and the ruff gate; notes, ACs.

## Implementation Notes

**Status.** AC#2 and AC#3 are met. AC#1 can only be read from the 10 PR runs after merge, so it stays open, and so does the task. On the slowest runner seen (the four runs cancelled at the cap), the projection below puts the head at up to about 18 min and the tail at up to about 25 min of 35: about 10 minutes under the cap, which meets AC#1's 8 with little to spare. Completed runs project lower (about 16 and 20 min).

**Owner decision.** 2026-11-24: asked about the split, the owner chose "Split, +1 runner (Recommended)".

**Evidence.**
- Window: 126 PR jobs, 2026-11-22 03:42Z to 2026-11-24 23:09Z.
- Admission-sensitive profile: 115 succeeded, at a median of 29.8, p90 of 33.6 and max of 34.8 min. Four more were cancelled at the 35-min cap with every test so far passing (45050043124, 45435409101, 45446280626, 45455465752; the last died in the discarded-tool-run file at 87%).
- The other profiles had plenty of room: sandboxed max 7.5 min, restore max 3.9.
- Per-file times came from nine CI logs. Each progress line is timestamped when its newline arrives, so a file's time is the gap between line ends. No single file dominates:
  - project-instruction traces: 303 s on average
  - failed tool run: 294 s
  - discarded tool run: 178 s
  - row-menu sweep: 127 s
  - skill-discovery capture: 115 s
  - the two wrapped-collaborator files: 105 s and 102 s
  - everything else: under 60 s each

**Choice: a split** (the owner's decision above).
- (c), fixing slow files, was rejected. Halving the largest file would save about 2.5 min, not the 7+ that AC#1 needs.
- (b), moving files to a step with headroom, was rejected. A bootstrap_profile file can only go to an admission-sensitive invocation. Dreams boundary has about 10 min left (successful runs max 25.2 of 35, 2026-11-24..25); the PR admission jobs reach 24-29 of 35 and are off limits anyway.
- Folding the tail into the restore profile was considered and rejected. It would add no runner, but TASK-40724 gave each invocation its own runner after the restore selectors once took 25m46s under sustained load.
- So `console-p0-gate` gets a fourth matrix profile, `admission-sensitive 2`. Its step, "Run the admission-sensitive 2 Console P0 regression tests", takes the old step's tail from `test_console_project_instruction_traces.py` on: 24 targets with the same order and flags (`--timeout=300 --tb=short`), and the PR3023 witnesses still last.
- The old step keeps its first 43 targets. The two lists concatenated equal the old list exactly, no file appears in both, and both are bootstrap-profile invocations, so isolation is unchanged.
- The job's `if`, its 35-min cap and the aggregate's `needs` and single verdict on `needs.console-p0-gate.result` cover every matrix profile unchanged.
- Cost: one more runner per full PR run, which makes 57 eligible test jobs, 36 concurrent runners and 37 with perf-guard. That is about 1.5 more runner-minutes per run (setup, collection and cleanup are about 85 s a job on completed runs, up to about 130 s on the slowest).

**Timings.**
- CI split of the nine completed logs, pytest time only:
  - head: median 846 s, max 880 s, about 16.1 min a job with about 85 s of setup, collection and cleanup
  - tail: median 1052 s, max 1115 s, about 20.0 min a job
- The four runs cancelled at the cap were slower, and they set the ceilings (setup with collection 85-127 s):

  | Run | Head | Tail | Setup |
  |---|---|---|---|
  | 45455465752 | 972 s | ~1400 s projected (killed in the discarded-tool-run file; project-instruction traces alone 521 s) | 92 s |
  | 45446280626 | 877 s | ~1265 s projected (killed in the agent-call thinking file) | 127 s |
  | 45435409101 | 866 s | ~1230 s projected (killed near the end) | 85 s |
  | 45050043124 | 902 s | ~1124 s | 85 s |

  - so a job takes up to about 18 min (head) and about 25 min (tail): about 10 min under the 35-min cap on the slowest runner seen
- Linux check: ubuntu:24.04, Python 3.12.3, SQLite 3.45.1, ext4 named volumes for /tmp and /home/runner, user runner uid 1001, the job's exact install line, network disconnected, `--cpus 4`.
  - unsplit step: 528 passed in 1393 s
  - head: 294 passed in 588 s
  - tail: 234 passed in 683 s
  - completed CI runs took 1.40x (median) to 1.46x (max) the unsplit Linux time, so for them the projection is about 14.7-15.3 min (head) and 16.9-17.6 min (tail) with setup; the factor comes from completed runs only, and 45455465752 ran at about 1.7x
  - the container, volumes and image were removed afterwards
- Local, serially via `r19-step.sh` with the step's flags:
  - head: 294 passed in 626 s (wall 634 s) at load 8-11
  - tail: 234 passed in 870 s (wall 876 s) at load 8-13
- AC#2: in a fresh process, the tail under the `RecoveryRequired.__init__` recorder (`$FU/probe-rr/r24rrcount.py`) passed 234 with **0** hits. No file moved into a sandboxed invocation.

**Contracts, and their red evidence.**
- `Tests/CI/test_response_rules_critical_gate.py`:
  - `SELECTION` gains an empty `admission-sensitive 2` label.
  - `INHERITED` splits the admission list and pins the new step token by token.
  - `STEP_JOBS` gains the new step, and the step count goes from 6 to 7.
  - The weakening checks now run against both admission steps.
  - `test_missing_or_comment_only_rule_target_is_rejected` is parametrized over (step name, target) pairs from `SELECTION` plus the new step's `INHERITED` targets, so every new-step target, removed or commented out, fails the contract; each case first asserts the target is executable in its step (review fix, 2026-11-25: folded in a duplicate test).
- `Tests/CI/test_console_p0_regression_gate.py`: the new lane is added to `_lanes()` and to the bootstrap-OK set, so the run-once and overlap checks see it.
- `Tests/CI/test_ci_queue_pressure_contract.py`: four profiles and seven steps; the new step's `if` and `--timeout=300`; no target shared with the PR admission jobs or with the head step; and a continue-on-error case.
- `Tests/CI/test_self_hosted_runner_routing.py`: the matrix and four pytest steps.
- `Tests/CI/test_derived_artifacts_workflow.py`: the step list, and the PR3023 step is now the new one.
- `Tests/CI/test_ui_lane_shard_count_statements.py`: docstring, and a 4-profile case.
- Red, edit in place then revert:
  - Commenting out `Tests/Chat/test_console_trace_model_retries.py` in the new step turned `test_rule_critical_targets_execute_without_altering_inherited_processes` red.
  - Deleting the PR3023 native-commit witness turned that test red, plus the PR3023, TASK-34975 and TASK-42342 order tests.
  - Before ADR-103 was updated, `test_ui_lane_shard_count_statements.py` was red three times (56 vs 57, 35 vs 36, 36 vs 37).

**Files.**
- `.github/workflows/derived-artifacts.yml`
- `Tests/CI/test_response_rules_critical_gate.py`
- `Tests/CI/test_console_p0_regression_gate.py`
- `Tests/CI/test_ci_queue_pressure_contract.py`
- `Tests/CI/test_self_hosted_runner_routing.py`
- `Tests/CI/test_derived_artifacts_workflow.py`
- `Tests/CI/test_ui_lane_shard_count_statements.py`
- `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`: counts, plus a dated amendment
- `scripts/ci/self_hosted/README.md`

---

**2026-11-28, re-measured (AC #1).** The split reached dev with PR 3782 (merged 2026-11-26 04:27Z). AC #1 asks that the slowest run end at least 8 minutes under the cap. **It does not: over the 72 jobs on the split list the slowest ended at 29.7 minutes, 5.3 under.** Nothing is applied. AC #2 and AC #3 stay met; AC #1 is not ticked, and rewording it to the 4-minute floor of TASK-34986 is the owner's to do, not a fit to the result.

**Awaiting the owner.** Which option below. The recommendation is option 2, three files out of the head. Raising a limit and adding a runner are both the owner's decision, so neither is applied.

**Checked first.** This task is the only one on any ref about this profile's cap. Console P0's block is edited by 13 remote refs against their merge base with `origin/dev`: PR 3345 (`codex/shared-console-mobile-pr`, draft) moves one file into `admission-sensitive 2` (TASK-42779's option 3) and changes the install line to add `pytest-mock`; PR 3572 (`codex/console-validation-performance`, 2,253 commits behind dev) adds two targets to a profile it does not name, "runner cost not yet measured"; the rest are `runs-on` or `if` lines, old restructures, or `codex/wave2-pr3023-*` branches with no open PR. No open pull request changes the cap.

**Measured.** GitHub API job records of the 258 `pull_request` runs of `derived-artifacts.yml` created from 2026-11-24 to 2026-11-28 17:42Z, each job execution counted once, and the logs of the 99 jobs of this profile that finished since 2026-11-26 00:00Z. None was cancelled at the cap (the one cancellation near it, job 140039303766 at 36.5 minutes, reads "canceled by @github-actions[bot]"). Ten of the 99 ran an older list before the split merged, between 00:22Z and 04:41Z (eight with 531 or 530 tests in 44 or 43 files, two with 296 tests: the split's own branches, head only). **The 34.2-minute job that makes this profile look at the cap since 2026-11-26 is one of the ten** (job 139663139270, run 45741860736, `fix/estop-holds-agent-tool-calls`, 03:36Z, 531 tests, Azure `eastus`). PR 3345's own 17 jobs ran a 485-test list. The 72 jobs on today's list (506 tests, 26 files):

| | Median | p90 | Slowest |
| --- | --- | --- | --- |
| observed, 72 jobs on today's list | 24.4 | 26.3 | 29.7 |
| modelled, all 99 runner speeds on today's list | 23.4 | 26.2 | 29.6 (5.4 under) |

Reference job 139928504976: 1,379 s of tests plus 81 s of setup and collection. Runner speed over the 99 jobs runs 0.70 to 1.22 times the reference (1.74 times; AWS 0.70 to 0.82, Azure 0.73 to 1.22). The model reproduces the 72 observed jobs (median 24.3, p90 26.2, max 29.6). A lognormal fit puts the share over the cap at 0.02%.

**Where the time goes** (reference log): `Chat/test_console_failed_call_retry_ownership.py` 589 s (43%, 210 tests; TASK-42525 put it last in the head (round 26, 2026-11-24 to 25), and its comment expected "about 24-27 of 35"), `UI/test_console_row_menu_action_sweep.py` 121 s, `Chat/test_console_send_wrapped_skill_collaborators.py` 96 s, `UI/test_console_skill_discovery_wrapped_collaborators.py` 93 s, `Chat/test_console_trace_rewritten_history.py` 54 s, `Chat/test_response_rules_write_contention.py` 49 s, `UI/test_console_received_route_wrapped_collaborators.py` 46 s, `UI/test_console_hook_review_send_freeze.py` 44 s, `UI/test_console_terminal_return_input.py` 39 s, the other 17 files 248 s. The notes above said the head would take up to about 18 minutes on the slowest runner; that was before the retry file joined it. The tail, `admission-sensitive 2`, ran median 17.2, p90 19.4, slowest 20.2 over 89 jobs (234 tests, 18 files, 955 s of tests): 14.9 minutes of room.

**Options** (the 99 runner speeds on today's list; "Under" is the slowest against the cap):

| | Change | Runners | Median | p90 | Slowest | Under | Over the cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | none (cap 35) | 0 | 23.4 | 26.2 | 29.6 | 5.4 | 0 of 99 |
| 1 | cap 35 to 40 | 0 | 23.4 | 26.2 | 29.6 | 10.4 | 0 |
| 2 | the three wrapped-collaborator files (96 + 93 + 46 = 235 s) to another bootstrap profile | 0 | 19.6 | 22.0 | 24.8 | 10.2 | 0 |
| 3 | the retry file (589 s) in a job of its own | +1 | 14.0 and 10.8 | 15.6 and 12.0 | 17.6 and 13.5 | 17.4 and 21.5 | 0 |

Destinations for option 2, from the same method: `admission-sensitive 2` with the 235 s ends median 21.0, p90 23.7, slowest 24.8 (10.2 under); with TASK-42779's workflow-planning file (315 s) instead, 26.3 (8.7 under); **with both, 31.0 (4.0 under)**, so it can hold one of the two moves at the 8-minute margin and not both. The `restore` profile (median 17.1, max 19.5 over 32 jobs since 2026-11-27 15:00Z, after its traced actions file joined it in PR 3797) would end about 24 with the 235 s (a job-level estimate, 10 under); its four files are pinned as the restore selectors plus that file. Both are bootstrap-profile invocations, so isolation is unchanged, and no sandboxed target may follow a bootstrap-profile one in either.

1. **Cap 40** meets AC #1's number (10.4) by raising the limit rather than by changing the work, and the profile has no cap hit to answer. No runner.
2. **Three files out.** No runner, no limit; takes the head to median 19.6 and 24.8 at the slowest. It needs the two Console P0 contract files (`test_response_rules_critical_gate.py`'s `SELECTION` and `INHERITED`, `test_console_p0_regression_gate.py`'s lanes) and a dated ADR-103 note, and only a full CI run proves a moved bootstrap-profile file.
3. **A job of its own for the retry file.** One more runner per PR (dev's tip counts 36 concurrent test runners, 37 with perf-guard; PR 3804's new job, the owner's pick of 2026-11-28, makes it 37 and 38; this one would make it 38 and 39, against the ceiling of 40 in ADR-103), about 1.5 more runner-minutes of setup per run. The owner chose a fourth profile for this job on 2026-11-24 ("Split, +1 runner").

**Recommendation: option 2.** It restores AC #1's margin with no runner and no limit. Its destination competes with TASK-42779's recommended move for `admission-sensitive 2`; the two together leave that profile 4.0 under, so send one of them to `restore` or to a new job. If the owner would rather keep the profile map as it is, the alternative is to read AC #1 as the 4-minute floor (5.4 met), which only the owner can decide.

**Not applied, and why.** Options 1 and 3 each need the owner (a limit or a runner); option 2 needs the owner's choice of destination, and a change under `.github/` starts the full CI run. The full table of lanes is in `backlog/docs/ci-lane-headroom-2026-11-28.md`.

**How it was measured.** As in TASK-42779's note (same-test-count files, speeds against a complete log of today's list, each cancellation's annotation read).

**2026-11-28, later: TASK-42779's choice changes option 2's destination.** The owner chose lane 4's one-file move (TASK-42779, option 3): `Tests/Chat/test_workflow_planning_native.py` (315 s) now runs first in `admission-sensitive 2`, which takes that profile from median 17.2 / slowest 20.2 to a modelled median 22.3 / slowest 26.3 (8.7 under). This task's three wrapped-collaborator files (235 s) would take it to 31.0 (4.0 under) if they went there too, so option 2's destination should now be the `restore` profile (about 24, a job-level estimate) or a job of its own. AC #1 stays open and unticked.
