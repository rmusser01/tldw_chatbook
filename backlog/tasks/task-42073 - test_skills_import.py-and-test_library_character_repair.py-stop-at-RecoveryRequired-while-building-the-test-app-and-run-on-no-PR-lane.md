---
id: TASK-42073
title: "test_skills_import.py and test_library_character_repair.py stop at RecoveryRequired while building the test app and run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 02:35
updated_date: 2026-11-20 17:43
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
- **Cause of the serial `assert 0 == 20` (AC#4).** It was a test race, not a product defect and not a swallowed exception. The dialog's `_start_refresh` runs an async worker (`_refresh_owned`) that awaits `asyncio.to_thread(load_refresh)`. The test read `controller.candidates` after one `pilot.pause()`, and that pause returns at the first 20 ms window in which the process looks CPU-idle (Textual `wait_for_idle(0)`). Evidence from temporary probes, since reverted:
  - The pause took 41-47 ms. It returned with `_refresh_owned` still RUNNING, the status still "Refreshing authoritative repair choices…", and 0 candidates.
  - The worker's first read on a fresh executor thread (`get_character_conversation_search_revision`, which opens that thread's SQLite connection) took 64-72 ms wall but only 24-28 ms CPU.
  - A probe in the `except` branch never fired.
  - After `app.workers.wait_for_complete()` there were 20 candidates.

  This is the TASK-41513 lesson's shape. **Correction to the task description:** on base the race is not serial-only. Three runs of the file at -n 4 each failed at least one continuation node (2, 3 and 2 failures counting the RecoveryRequired node), so "passed at -n 4" was luck.
- **Fix.** The test now awaits `app.workers.wait_for_complete()` after each pause that follows a worker-starting action: the push, the two Next presses, the Refresh, and the restart Next. No assertion changed.
- **Negative control.** With `_next_page` made a no-op (a temporary edit, since reverted), both continuation nodes went red at `assert 20 == 40`, serially and at -n 4.
- **The mounted-Library node after the marker (AC#3).** It was the only repair test still failing, and it is fixed here. It had two failures of the same race class:
  - `assert not select.disabled` read the dialog before its refresh worker landed. It now waits for the dialog's workers first.
  - In 1 of 8 serial runs, `pending_repair_context is None` failed. `wait_for_complete()` ran straight after the confirm `press()`, before the press had started the CAS worker, so the app exited mid-repair. A `pilot.pause()` now comes between the press and the wait.
  - Review rev1 asked for the same pause in the two other press-then-wait sites: `test_candidate_change_revokes_armed_repair_in_real_dialog` and `test_cas_exception_restores_retry_controls`.
- **Skills import helper race (review rev1).** `_run_skills_import_via_ui` returned on the first status change, which can be the in-flight "Inspecting/importing…" line. The reviewer saw `test_import_real_superpowers_skills_lands_trust_pending` fail that way. The helper now returns only a change painted after `screen._library_skills_import_in_flight` has cleared, and never the in-flight line itself.
  - Red: the theme-node-first order passed once here (50 passed, 451 s at load 29-33). With a temporary 0.5 s `asyncio.sleep` at the top of the coordinator's `_import`, the old helper failed `test_import_real_superpowers_skills_lands_trust_pending` with `'Inspecting/importing…' == 'Imported "executing-plans" …'`, and `test_import_row_rejects_name_too_long_without_partial_state` the same way.
  - Green: with the delay still in and the fixed helper, the whole file passed (49 passed). The delay is reverted.
- **Earlier failing serial Skills runs.** Two runs failed before the disk ran out: one with 5 failures, one with 1, at 15:4x UTC with 1.2-1.6 GB free. Both were reported as failures, not setup errors, and every failing node calls `_run_skills_import_via_ui`, so they were most likely this helper race rather than the disk. The runs that did hit the disk, between 15:44 and 16:00 UTC (Container Free fell to 123-133 MB), failed at setup with `OSError: could not create numbered dir`, `[Errno 28] No space left on device` and `sqlite3.OperationalError: disk I/O error`.
- **Lanes (AC#2).**
  - **UI census:** `Tests/UI/test_library_character_repair.py` is appended at the census tail. Census charge (15 passed each run):
    - 10 serial runs: 10.9-11.9 s at load 10-11;
    - 3 runs with `-p no:xdist --timeout=180 --durations=0`: 14.0, 14.3 and 16.2 s at load 11-12.
  - **Skills import file:** goes *first* in `pr-fast-lane-admission-3`. Every target in that lane is bootstrap_profile, so order is free under TASK-32873. The `ADMISSION_3_TARGETS` pin is updated to the same order.
    - Placed last, the file took 497 s in the reviewer's lane-order run, 383 s of it teardown.
    - Placed first, the whole step took 780 s locally (serial, the lane's flags, load 17-21, 277 passed). The Skills file took 88.5 s of that (call 78.6 s, teardown 9.6 s); the Roleplay import app 160 s, the item list 472 s, the theme node 56 s.
    - That leaves the lane far inside its 35-minute cap less the 5.3-minute hung-test margin (CI ran this lane at up to 20 of 35 min before this change).
    - Serial with the UI lane's flags, before review rev1: 55.0, 54.0 and 58.3 s (49 passed each) at load 3.9-6.2.
  - **Teardown blow-up, named.** After the theme-contrast node the process holds ~10.6M gc-tracked objects (373k before it). Most of them are parsed CSS: 516k `RuleSet`, 1.1M `Selector` and 515k `Styles`. They sit in `Tests/UI/css_cache.py`'s unbounded session-wide `_GLOBAL_PARSE_CACHE`, which keys on the stylesheet variables, so the node's 89 themes add 89 parses of the app stylesheet. `Tests/conftest.py::cleanup_file_descriptors` runs a full `gc.collect()` after every test under an app-mounting directory, Tests/Skills included, and each collect now takes ~5.5 s. Control: with `TLDW_TEST_CSS_CACHE=0`, the theme node's teardown fell from 5.8 to 0.5 s and the next Skills tests' teardowns from 5.3-5.4 to 0.11 s. Bounding that cache is left as a proposed follow-up.
- **Verification after review rev1.**
  - Skills file, serial with the UI lane's flags: 49 passed in 106.8, 88.4 and 70.2 s at load 21-38.
  - Repair file, serial: 15 passed in 13.2, 12.9 and 12.7 s at load 14.
  - Both files at -n 4: 64 passed.
  - Before review rev1, the repair file also passed 13 of 13 serial runs and 3 of 3 at -n 4.
  - Preflight passed. `Tests/CI` passed in full (1811 passed, 2 skipped). Ruff F and `ruff_gate.py` passed.

**Files:** `Tests/Skills/test_skills_import.py`, `Tests/UI/test_library_character_repair.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
