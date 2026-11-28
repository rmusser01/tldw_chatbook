---
id: TASK-42533
title: "The admission-sensitive Console P0 job runs 27-34 minutes of its 35-minute cap"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 09:14
updated_date: 2026-11-28 18:06
labels:
- follow-up
- ci
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
- [x] #1 Over the next 10 PR runs after the change, the admission-sensitive profile's slowest run ends at least 8 minutes under its cap
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

**Status.** All three criteria are met; AC#1 was read from the PR runs after merge (see **AC#1, measured** at the end). Before merge, on the slowest runner seen (the four runs cancelled at the cap), the projection below put the head at up to about 18 min and the tail at up to about 25 min of 35: about 10 minutes under the cap, which meets AC#1's 8 with little to spare. Completed runs project lower (about 16 and 20 min).

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

**AC#1, measured (2026-11-28).** The split merged as #3782 at 2026-11-26 04:27Z. Job times are read from GitHub's jobs API for the derived-artifacts PR runs since then (119 runs; five of them answered HTTP 502 at first and were re-read; completed jobs only, success or failure). A cancelled job is left out: none ran to the cap, and the long cancelled ones (29.0, 36.5 and 47.7 min) never got a runner (runner_id 0, no steps).
- **The first 10 PR runs** (2026-11-26 04:29Z to 08:11Z): "admission-sensitive" was slowest at 26.8 min, 8.2 under the 35-minute cap, so AC#1 holds; "admission-sensitive 2" was slowest at 20.2.
- **All 85 completed runs of each:**
  - "admission-sensitive": median 22.9, p90 26.3, max 29.7 min. Three runs went above 27: 27.6, 29.7 and 27.1.
  - "admission-sensitive 2": median 17.2, p90 19.4, max 20.2.
- **Against the projection:** the projection was not wrong; the head grew after it. #3784 (TASK-42525) merged at 2026-11-26 04:28:20Z, 21 s after the split and before the first measured run. It moved `Tests/Chat/test_console_failed_call_retry_ownership.py` from the Dreams tracking riders to the end of the head step and grew it by about 2470 lines. That file takes about 10-12 min in CI (620 s and 723 s in runs 45763004776 and 45948611554). Without it, the head matches the projection. Moving about 3 min from the head to the tail would balance them; TASK-42652 tracks that.

