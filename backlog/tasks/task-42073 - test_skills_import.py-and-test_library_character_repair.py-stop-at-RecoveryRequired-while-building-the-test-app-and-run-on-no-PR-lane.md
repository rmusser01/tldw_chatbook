---
id: TASK-42073
title: "test_skills_import.py and test_library_character_repair.py stop at RecoveryRequired while building the test app and run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 02:35
updated_date: 2026-11-20 16:16
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/Skills/test_skills_import.py
- Tests/UI/test_library_character_repair.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41715 (round 17) when its touched test files were run on the restacked branch. 25 nodes in `Tests/Skills/test_skills_import.py` and 1 in `Tests/UI/test_library_character_repair.py` (`test_mounted_library_refreshes_real_cas_version_then_retry_succeeds`) fail with `RecoveryRequired: raw_source_selection_changed` while building the test app: `_build_test_app` -> `load_settings` -> `config_participants.py` -> `Backup_Recovery/raw_participants.py`. This happens on dev e446ab06eb and on dev a6df4d8d48, with or without round 17.

Two more nodes in `test_library_character_repair.py`, `test_repair_continuation_reaches_every_candidate_and_restarts_after_mutation[size0|size1]`, fail for another reason. They build a bare `App()` with their own database, so no profile marker applies, and assert `len(controller.candidates) == 20` after a single `pilot.pause()`. Run serially (alone or with the file) they fail with `assert 0 == 20` every time (5 of 5 across two trees); at `-n 4` they have passed. The cause is not known: the dialog loads candidates in a worker (`_refresh_owned`, `asyncio.to_thread`) whose exceptions only set the status line to "Refresh failed. Retry or cancel." and are not logged, so a swallowed load failure is not ruled out.

Neither file carries the `bootstrap_profile` or `@private_profile_test` marker that app-building census files use, and no PR lane runs either file. That is the same shape as TASK-41772, whose scope is 63 named media-series guards and does not include these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every test in both files that builds the app gets the profile setup it needs and no longer stops at RecoveryRequired
- [x] #2 Each of those tests that then passes 3 of 3 serially with the UI lane's flags runs on a PR lane
- [x] #3 Any test that still fails after the setup fix has its failure recorded, with its own task or fix named
- [x] #4 The cause of the two repair-continuation nodes' serial `assert 0 == 20` is named, and both pass 3 of 3 serially and at -n 4 with their assertions unchanged
<!-- AC:END -->

## Implementation Plan

1. Reproduce both files' failures serially and at -n 4 on the base (through the scratch wrapper).
2. Profile setup: follow the Library harness's own pattern (`Tests/UI/test_library_shell.py` carries a module `pytestmark = pytest.mark.bootstrap_profile`, and `_build_test_app` reads config). Give `Tests/Skills/test_skills_import.py` the module marker (every test there builds the Library app); give the one mounted-Library node in `test_library_character_repair.py` the per-node marker.
3. AC#4: instrument `_refresh_owned` temporarily (or read the dialog status) to find why the candidate load leaves 0 candidates after one pause when serial; fix the root (test timing or product) without changing the assertions.
4. Rerun each file 3x serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`) and at -n 4; record times and load.
5. Lanes: append the Tests/UI file to the UI census tail (bump MINIMUM_FILES); add Tests/Skills/test_skills_import.py as a whole-file target in an existing non-UI step of derived-artifacts.yml; update Tests/CI pins that trip.
6. Preflight, notes, Done.

## Implementation Notes

**Approach.** Test-only changes plus two lane entries; no product code changed.

- **Profile setup (AC#1).** `Tests/Skills/test_skills_import.py` gets a module `pytestmark = pytest.mark.bootstrap_profile`, the same as `Tests/UI/test_library_shell.py`, whose `_build_test_app` it uses. All 25 failing nodes build the app through that factory. The other 24 are unit-level and pass either way. `test_library_character_repair.py` builds the app in one node only, so that node gets the per-node marker (TASK-32873's per-node opt-in, as `test_library_skills_canvas.py` does). The markers live in the files rather than in `Tests/conftest.py`'s name list. The RecoveryRequired errors are gone: the skills file went from 25 failed / 24 passed at -n 4 on base to 49 passed.
- **Cause of the serial `assert 0 == 20` (AC#4).** It was a test race, not a product defect and not a swallowed exception. The dialog loads candidates in `_refresh_owned` through `asyncio.to_thread(load_refresh)`. The test read `controller.candidates` after one `pilot.pause()`, and that pause returns at the first 20 ms window in which the process looks CPU-idle (Textual `wait_for_idle(0)`). Evidence from temporary probes, since reverted:
  - The pause took 41-47 ms. It returned with `_refresh_owned` still RUNNING, the status still "Refreshing authoritative repair choices…", and 0 candidates.
  - The worker's first read on a fresh executor thread (`get_character_conversation_search_revision`, which opens that thread's SQLite connection) took 64-72 ms wall but only 24-28 ms CPU.
  - A probe in the `except` branch never fired.
  - After `app.workers.wait_for_complete()` there were 20 candidates.

  This is the TASK-41513 lesson's shape. **Correction to the task description:** on base the race is not serial-only. Three runs of the file at -n 4 each failed at least one continuation node (2, 3 and 2 failures counting the RecoveryRequired node).
- **Fix.** The test now awaits `app.workers.wait_for_complete()` after each pause that follows a worker-starting action: the push, the two Next presses, the Refresh, and the restart Next. No assertion changed.
- **The mounted-Library node after the marker (AC#3).** It was the only test still failing, and it is fixed here. It had two failures of the same race class:
  - `assert not select.disabled` read the dialog before its refresh worker landed. It now waits for the dialog's workers first.
  - In 1 of 8 serial runs, `pending_repair_context is None` failed. `wait_for_complete()` ran straight after the confirm `press()`, before the press had started the CAS worker, so the app exited mid-repair. A `pilot.pause()` now comes between the press and the wait.

  No test is left failing, so no follow-up task is needed.
- **Negative control.** With `_next_page` made a no-op (a temporary edit, since reverted), both continuation nodes went red at `assert 20 == 40`, serially and at -n 4.
- **Lanes (AC#2).**
  - `Tests/UI/test_library_character_repair.py` is appended at the UI census tail, and `MINIMUM_FILES` is now 2440. Census charge: 15 passed in 10.9-11.9 s in 10 serial runs at load 10-11. In 3 later runs with `-p no:xdist --timeout=180 --durations=0` it took 14.0, 14.3 and 16.2 s at load 11-12.
  - `Tests/Skills/test_skills_import.py` is appended as a whole-file target to `pr-fast-lane-admission-3`, the bootstrap_profile-only lane with the most headroom (20 of 35 min per its 11-18 note). The matching `ADMISSION_3_TARGETS` pin in `Tests/CI/test_ci_queue_pressure_contract.py` is updated. Serial time with the UI lane's flags: 55.0, 54.0 and 58.3 s (49 passed each) at load 3.9-6.2 with 11.2-11.6 GB free.
- **Runs lost to the disk.** Earlier serial skills runs failed while the disk ran out (07:44-08:00Z, Container Free fell to 123-133 MB). The captured runs failed with `OSError: could not create numbered dir`, `[Errno 28] No space left on device` and `sqlite3.OperationalError: disk I/O error`. Two earlier failing runs (5 and 1 failures) were not captured. They fell in the same window, with free space falling from 1.6 GB, but their cause is not proven.
- **Verification.**
  - Both files at -n 4: 64 passed.
  - The repair file at -n 4: 15 passed in each of 3 runs.
  - Serially, the repair file passed 13 of 13 runs after the fix.
  - Preflight passed. `Tests/CI` passed in full (1811 passed, 2 skipped). Ruff F and `ruff_gate.py` passed.

**Files:** `Tests/Skills/test_skills_import.py`, `Tests/UI/test_library_character_repair.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

