---
id: TASK-34929
title: The Dreams tracking tests run in no pull-request lane
status: Done
created_date: 2026-10-12 23:18
labels:
- ci
- tests
- dreams
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-14 15:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Dreams/test_track_service.py, about 200 tests of the Dreams tracking and switch logic, is in no group the required check runs; only the scheduled full-tree workflow runs it. Seven of its tests went red with the PR #2921 merge and stayed red unnoticed (fixed in the follow-up PR to #2921). One of its tests, test_overlapping_reconciles_report_settled_once_at_the_end, also waits on fixed 5 s and 10 s bounds and runs five times slower under load. The PR Fast Lane has almost no time left (TASK-34916), so adding the file needs room made first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Dreams tracking tests run on every pull request that changes Dreams or Subscriptions code
- [x] #2 No required job comes closer to its time cap as a result
- [x] #3 The overlapping-reconciles test waits on events rather than fixed bounds, or its bound has a measured margin under load
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the file's serial cost locally and calibrate against CI with Dreams-family files that already run in the P0 gate.
2. Give it a lane that runs on every pull request without touching any existing job's time: its own job in derived-artifacts.yml, aggregated by the required check.
3. Replace the overlapping-reconciles test's fixed 5 s / 10 s wait_for bounds with plain awaits on the events and tasks (pytest-timeout still bounds a hang); same change for the two sibling tests using the same parked-source bounds.
4. Pin the new job in the contract tests; verify the file, Tests/CI and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Gave Tests/Dreams/test_track_service.py its own required lane and took the fixed time bounds out of its parked-source tests.

Lane: new job dreams-tracking ('Dreams tracking tests') in derived-artifacts.yml runs the whole file on every pull request (and on full-gate dispatches), serial, same minimal install line, --timeout=180, 20-min cap; the required aggregate needs it with a 'Require successful Dreams tracking tests' verdict. No existing job's steps changed, so no required job moves toward its cap (AC#2).

Why its own job: the file is 219 tests and measured 6m35s and 5m42s serially here (load average 10-18). Calibration against CI: four Dreams-family files that already run in console-p0-gate took 1.3-1.7x their local time there (test_dispatch_currentness 7.8 s local / 13.0 s CI, test_dreams_activation 11.9 / 15.9, test_local_watchlists_service 24.2 / 36.9, test_watchlists_bulk_source_authoring 59.9 / 80.8), so ~8-11 min in CI, plus ~1 min setup. That fits in no existing lane: after TASK-34916 the PR Fast Lane halves replay at max 20.8 / 20.1 of 35 (adding it would break 34916's 5-minute margin), console-p0-gate is at 18.7 of 20, the UI census admits only Tests/UI.

Why not path-filtered: GitHub has no job-level path filter, a skipped job fails the aggregate's != 'success' verdicts, and the file imports Notifications, Scheduling and Utils/egress as well as Dreams/Subscriptions, so a path list would miss the breaks it exists to catch. Running on every PR covers 'every PR that changes Dreams or Subscriptions code' (AC#1). Cost: one more runner (~10-12 min) per PR run; it finishes before the PR Fast Lane, so it adds no wall-clock. A step-level path gate inside this job is the lever if the runner minutes ever matter.

Test (AC#3): test_overlapping_reconciles_report_settled_once_at_the_end and its two siblings using the same _ParkedSources helper with the same 5 s / 10 s wait_for bounds (test_overlapping_reconciles_settle_on_the_latest_switch, test_cancelling_a_reconcile_caller_does_not_cut_the_switch_short) now await the parked event and the reconcile tasks directly; a hang is bounded by pytest-timeout, which also dumps the stacks. These were the only wait_for calls in the file. Negative controls: running on_settled between coalesced passes in _reconcile_until_settled fails the test; never releasing the parked source fails it with 'Timeout (>15.0s) from pytest-timeout' (--timeout=15) instead of hanging. Production code restored byte-identical afterwards.

Verification: the whole file 219 passed serially (341.8 s) after the change; Tests/CI + Tests/Packaging/test_python_runtime_floor.py 915 passed, 1 skipped; negative controls for the new contract (target removed, the file also listed in another step, job dropped from needs, a -k filter) are all caught; ruff findings on the touched test files are identical per file and message to dev; ./scripts/preflight.sh passed. The job's first real run is the PR's own CI.

Files: .github/workflows/derived-artifacts.yml, Tests/Dreams/test_track_service.py, Tests/CI/test_ci_queue_pressure_contract.py, Tests/CI/test_derived_artifacts_workflow.py, Tests/CI/test_console_p0_regression_gate.py, backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
