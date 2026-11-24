---
id: TASK-42480
title: "Lane Tests/test_keyring_isolation.py: the suite's keyring-isolation guard runs in no pull-request lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 11:13
updated_date: 2026-11-23 23:12
labels:
- follow-up
- ci
- tests
priority: low
references:
- Tests/test_keyring_isolation.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42477, whose guard tests had to go into `Tests/test_network_guard.py` because their natural home is not laned. Checked on `fu/r22-keychain` (62531d037a plus TASK-42477).

**Why.** `Tests/test_keyring_isolation.py` pins that the suite never reaches the developer's real OS credential store (TASK-19570 A). It checks three things: the bootstrap's null `keyring` backend, the production credential seam refusing it, and a constructed app recording the store as unavailable. No step in `.github/workflows/derived-artifacts.yml` names the file. It runs in no PR lane: only in `test.yml` (its `all-tests` and `core-tests` jobs; manual `workflow_dispatch`, or called from release qualification and publishing) and the scheduled `nightly-deep.yml` pass, which is no merge gate and usually reaches its 240-minute ceiling before root-level files. A pull request that removed the bootstrap's backend pin would merge with every lane green.

The natural lane is the "Run Architecture guard contracts" step beside `Tests/test_network_guard.py`; `Tests/CI/test_ci_queue_pressure_contract.py` pins that step's target list. Locally (2026-11-23, through the scratch wrapper), `test_a_constructed_app_records_the_store_as_unavailable` fails with the known local trap `RecoveryRequired: raw_source_selection_changed`. The other two pass. Check the app-building case in the lane before relying on it.

Coordinate with the other session's lanes work (TASK-42281), which edits `derived-artifacts.yml` and that contract file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Tests/test_keyring_isolation.py` runs in a pull-request lane, and the CI contract that lists the lane's targets names it
- [x] #2 All three of its tests pass in that lane, including the constructed-app case
- [x] #3 Removing the conftest's null-backend pin turns that lane red (shown once under a temporary edit)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Make the constructed-app case runnable in a lane: it fails locally (and, per the TASK-34982 lesson, in CI) with `RecoveryRequired: raw_source_selection_changed` because `load_settings()` runs under the per-test profile redirect. Run it under `@private_profile_test` (the `Tests/App/test_startup_init_hygiene.py` precedent for app-building tests outside Tests/UI); keep its assertions.
2. Add the file to the "Run Architecture guard contracts" step right after `Tests/test_network_guard.py` (sandboxed block; not a tail of pr-fast-lane / admission-4), and to `ARCHITECTURE_GUARD_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`, with a note in the job's header comment.
3. Time the file serially with the step's flags (`-p no:xdist --timeout=180 --durations=0`) and record the load; run the whole step through `r19-step.sh`.
4. Negative control (AC #3): comment out the conftest null-backend pin and rerun the step once, with the ambient backend set to `keyring.backends.fail.Keyring` so nothing can resolve the macOS Keychain (an unset backend on macOS is the login Keychain); revert.
5. Tests/CI, Architecture (serial), preflight, ruff gates; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Lane.** `Tests/test_keyring_isolation.py` is now a target of the "Run Architecture guard contracts" step (job `architecture-guards`, every PR), right after `Tests/test_network_guard.py` in the sandboxed block, before the ChaChaNotesDB/Tests/DB bootstrap-profile files. `ARCHITECTURE_GUARD_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` names it in the same place, so the contract pins it. Neither the pr-fast-lane nor the admission-4 tail was touched.

**The constructed-app case.** It failed with `RecoveryRequired: raw_source_selection_changed`: `_build_test_app` calls `load_settings()` under the per-test profile redirect, which config-participant admission refuses (the TASK-34982 lesson: the same happens in CI, so it would have been red in the lane). It now runs under `@private_profile_test` (with `@pytest.mark.asyncio` and `request`), the precedent `Tests/App/test_startup_init_hygiene.py` uses for app-building tests outside Tests/UI. Its assertions are unchanged. The child inherits this process's backend and its own conftest pins the same one.

**Timing** (the step's flags, serial `-p no:xdist --timeout=180 --durations=0`): 3 passed in 10.4 s at load ~45 (the app case 8.7 s call); 10.9 s for that case inside the whole step.

**Lane run.** `r19-step.sh fu-r24-misc "Run Architecture guard contracts" ... -p no:xdist --timeout=180`: 49 targets, 809 passed, 1 skipped (a Windows-only network-guard case), rc 0, 24:05 at load 40-75 (CI took 6-11 min of the job's 20 before this).

**AC #3, negative control** (temporary edit, reverted): with the conftest's `PYTHON_KEYRING_BACKEND` pin deleted, the file went red: `test_the_ambient_keyring_backend_is_the_null_backend` failed (`assert 'keyring.backends.fail.Keyring' == 'keyring.backends.null.Keyring'`), 1 failed, 2 passed, pytest rc 1, so the step fails. The run set the ambient backend to `keyring.backends.fail.Keyring` instead of leaving it unset: on macOS an unset backend resolves to the login Keychain, which the app case would then reach. On a Linux runner the same deletion leaves the variable unset and fails the same first assertion.

Files: `Tests/test_keyring_isolation.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
<!-- SECTION:NOTES:END -->
