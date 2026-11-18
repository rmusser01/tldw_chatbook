---
id: TASK-41516
title: "The 48 s TASK-41123 node sits in the serial pr-fast-lane"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:39
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/Backup_Recovery
dependencies:
- TASK-41123
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by bkreview (round 15). The TASK-41123 node takes 48 s and runs in the serial `pr-fast-lane`, which charges every PR. Moving it to a parallel lane keeps the fast lane fast.

Also: `Tests/Chat/test_fleet_wake_retry.py` is named by no PR lane; TASK-41138 only marked it (there is no `Tests/UI/test_fleet_wake_retry.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The node runs on a lane that is not the serial pr-fast-lane, with before and after fast-lane times recorded
- [x] #2 The node is still run by at least one PR lane
- [x] #3 `Tests/Chat/test_fleet_wake_retry.py` runs on a PR lane, with its time recorded
<!-- AC:END -->

## Implementation Plan

1. Read the PR lanes and the CI contracts that pin their targets.
2. Move `Tests/Backup_Recovery/test_skills_recovery_controls.py` (with the TASK-41123 node) from pr-fast-lane into the recovered-suites job's sandboxed block, and lane `Tests/Chat/test_fleet_wake_retry.py` there beside the other private-profile fleet-wake nodes.
3. Update FAST_LANE_TARGETS and RECOVERED_SUITES_TARGETS.
4. Time both files under the recovered-suites step's flags, and record the fast lane's time before and after.

## Implementation Notes

**Change.** `Tests/Backup_Recovery/test_skills_recovery_controls.py`, which holds TASK-41123's node, moves whole from pr-fast-lane's "Run fast PR contract" step into the recovered-suites job. It goes in that job's sandboxed block, after TASK-41521's persona coordinator file. `Tests/Chat/test_fleet_wake_retry.py` joins the same job after the three private-profile fleet-wake nodes (owner decision 2026-11-03). `FAST_LANE_TARGETS` and `RECOVERED_SUITES_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` follow. Recovered-suites is its own job, so it runs beside pr-fast-lane, and it is in the required aggregate's `needs`.

The whole file moves, not only the node. The recovered-suites contract refuses a node whose file another step lists whole, so moving one node would mean listing the file's other 13 nodes one by one in pr-fast-lane. Moving the file is a one-line change on each side, and it takes the file's other ~58 s out of the 35-minute fast lane too.

**Times.**
- Fast lane before, in CI: PR #3597's run 44198858833 at a9ef6290e1, the latest run that includes the node. The step took 1828 s (30:28, 4410 passed) and the job 31.6 of its 35 min. The file took 78.5 s of that (01:59:40.9 -> 02:00:59.4 in the log). TASK-41123's node took 20.7 s, the slowest in the step.
- Fast lane after: about 1750 s (4396 tests), projected as the step minus the file's 78.5 s. That is a projection, not a measurement. AC#1 stays unticked, and the task stays In Progress, until the coordinator reads the measured time from this branch's PR run and ticks it (review r16cs-rev1, Minor 2). Locally, the file alone with the fast lane's flags (`-p no:xdist --timeout=180 --tb=short`) took 70.0 s: 14 passed, the node 17.9 s, load 14-22.
- Recovered suites before: 660.7 s (job 11.9 of 20 min) in the same CI run. Under this step's flags (`--storage-lease-guard --timeout=180 --tb=short`, serial), locally: the skills file took 78.7 s (14 passed, the node 22.1 s, load 14-18), and `test_fleet_wake_retry.py` took 18.5 s (3 passed, 5.1-5.9 s each, load 14-17). That puts the step at about 760 s.
- The full fast lane was not timed locally. On this machine the load moved between 7 and 50 within minutes, and that noise over a 30-minute serial run is larger than the 78 s removed.

**Checks.** `Tests/CI`: with the base's missing persona-coordinator line added temporarily to `RECOVERED_SUITES_TARGETS` (see below), 1660 passed, 1 skipped, and 1 failed. The failure is the backup-owner contract, which also fails at the base. The lane target checks (`test_every_pr_lane_pytest_step_has_disjoint_targets`, `test_every_node_id_target_in_a_pr_lane_step_resolves`) are in that run.

**Base gaps found (TASK-41521's branch, not this task's).** At the base, `Tests/CI/test_ci_queue_pressure_contract.py` fails 2 tests:
- `test_recovered_suites_lane_is_one_serial_guarded_minimal_python_312_job`: 35de45b226 put `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py` in the workflow but not in `RECOVERED_SUITES_TARGETS`.
- `test_backup_owner_suites_run_on_dev_pushes_only_and_are_never_required`: ba0f910e01 added `test_created_owner_rollback.py` to the backup-owner job but not to `BACKUP_OWNER_SUITE_TARGETS`.

Both stay red on this branch until TASK-41521's branch adds those lines.

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
