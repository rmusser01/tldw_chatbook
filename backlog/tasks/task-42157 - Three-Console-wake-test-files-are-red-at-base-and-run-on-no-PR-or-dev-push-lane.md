---
id: TASK-42157
title: "Three Console wake test files are red at base and run on no PR or dev-push lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 22:23
updated_date: 2026-11-21 18:32
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_headless_wake_invariants.py
- Tests/Chat/test_console_fleet_wake_view_mark.py
- Tests/Chat/test_console_fleet_wake_staleness.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A consolerun follow-up (round 18), confirmed on fu/r18-petsdef, which does not touch these files (same command as dev, `-n 4`): 16 failed and 3 passed.
- `Tests/Chat/test_console_headless_wake_invariants.py`: 11 of 13 failed.
- `test_console_fleet_wake_view_mark.py`: 4 of 4 failed.
- `test_console_fleet_wake_staleness.py`: 1 of 2 failed.

None of the three files runs on a PR or dev-push lane (no entry in `.github/workflows/*.yml` or `scripts/ui_pr_gate_census.txt`). They do run in Core Tests (`.github/workflows/test.yml:132`, `pytest Tests --ignore=Tests/UI`, sharded, `--maxfail=1`), which runs at release through `workflow_call` and on manual dispatch. Whether they are red in that Linux job is not known.

The visible failures are mostly harness assertions: "the wake never delivered", "harness precondition: the manual send must run", and "the wake must be IN FLIGHT before quit". One raises `RecoveryRequired: raw_source_selection_changed` directly. Consolerun attributes them to the known local RecoveryRequired trap (plain tests outside the bootstrap-profile keep list). That makes them candidates for that keep list in `Tests/conftest.py`, but the cause is not yet shown for each test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each failing test's cause is shown (for example by running it under a temporary @private_profile_test or on the bootstrap-profile keep list); tests that fail for another reason are named
- [x] #2 The three files pass locally at -n 4, fixed or kept by the right mechanism
- [x] #3 The three files run on a PR or dev-push lane, with their times recorded
- [x] #4 The notes say whether these files are red in the Core Tests Linux job (test.yml, release and dispatch), from that job's log
<!-- AC:END -->

## Implementation Plan

1. Show each failing test's cause: probe every `RecoveryRequired` construction at base, and run the files on the bootstrap profile.
2. Apply the mechanism the sibling files use (module `pytestmark = pytest.mark.bootstrap_profile`, as `test_console_viewless_hooks.py` does).
3. Lane the three files whole in a bootstrap_profile PR lane (admission-sensitive 4) and its Tests/CI target list, with their serial times.
4. Read the Core Tests (test.yml) job logs and JSON reports for these files.

## Implementation Notes

All 16 failures had one cause, the local profile trap. The three files now keep the collection-time profile, the mechanism their sibling `test_console_viewless_hooks.py` uses, and run whole in PR Fast Lane (admission-sensitive 4).

**Cause, per test (AC#1).** At base dc371f0842, `-n 4`: 16 failed, 3 passed, as reported.
- A temporary `__init__` on `RecoveryRequired` (`Backup_Recovery/bootstrap.py`, reverted) printed every construction with its caller chain. Each of the 16 failing tests raised `RecoveryRequired('raw_source_selection_changed')` 9 to 14 times, all from `_register_raw_participant` under a storage admission `operation(...)`.
  - Two of them (`test_a_headless_wake_resolves_with_the_same_selection_as_a_manual_send`, `test_a_headless_wake_takes_the_same_agent_dispatch_and_budget`) also raised it under `_config_write_lock`. A second temporary probe at the two broad `except` blocks in `console_automatic_submission.py` showed their "Hooks unavailable" refusal is that same exception, from `read_hooks_config_snapshot()`.
  - Two (`test_navigation_detach_preserves_the_round_then_wake_delivery`, `test_the_global_cap_defers_a_headless_wake_like_any_other_send`) surface it directly. The other twelve swallow it in the wake path and fail on a harness assertion ("the wake never delivered", "must be IN FLIGHT before quit").
- With a temporary module `pytestmark = pytest.mark.bootstrap_profile` on each file, all 19 passed. No test fails for another reason.

**Fix (AC#2).** Each file gets a module `pytestmark = pytest.mark.bootstrap_profile` with a comment naming the cause. `-n 4`: 19 passed in 20.5 s (load 32).

**Lane (AC#3).** Appended to the `admission-sensitive 4` step in `.github/workflows/derived-artifacts.yml` and to `ADMISSION_4_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`. It is a bootstrap_profile lane, so the files never share a process with a sandboxed suite. Measured serially with that step's flags (`--timeout=300 --tb=short`, no xdist), load 32-36: 19 passed in 52.7 s. Per file: headless-wake invariants 34.5 s, view mark 9.5 s, staleness 3.1 s. The job's recent PR runs took 19-27 of its 35 minutes. `test_ci_queue_pressure_contract.py`: 153 passed.

**Core Tests (AC#4).** The only recent Core Tests runs are the two `publish-pypi.yml` pushes to main on 2026-11-19 (44507416213 at 8b89675ca2 and 44549304009 at 28391c524c); `test.yml`'s own dispatches since September skipped the Core job. Neither run reached these files, so the logs cannot say whether they are red on Linux.
- 44549304009: shards 0 and 4 stopped at collection (`Different tests were collected between gw1 and gw0`) with no test run. Shards 1, 2 and 5 ran 3,495, 3,780 and 2,911 of about 14,200 collected before their workers went down ("node down: Not properly terminated"). Shard 3 was cancelled.
- 44507416213: shards 1, 2, 3 and 5 stopped at collection with the same message. Shard 0 ran 2,461 tests before a worker went down. Shard 4 left no report.
- None of the 19 nodes appears in any shard's JSON report (`core-test-results-<n>`).

**Files:** the three test files, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
