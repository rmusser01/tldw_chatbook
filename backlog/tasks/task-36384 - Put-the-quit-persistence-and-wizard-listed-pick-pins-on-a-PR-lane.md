---
id: TASK-36384
title: Put the quit-persistence and wizard listed-pick pins on a PR lane
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 18:28
labels:
- follow-up
- ci
- tests
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/test_config_persistence_owner.py
- Tests/ProductionApp/test_config_profile_isolation.py
- Tests/Wizards/test_first_run_setup_wizard.py
- Tests/Wizards/test_first_run_setup_integration.py
dependencies:
- TASK-35578
- TASK-35835
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-35835 (PR #3216) and TASK-35578 (PR #3214) add pins that no PR lane runs. They run only on main pushes and nightly, so a PR that breaks them merges green. The UI census carries only `Tests/UI` nodes, and these live elsewhere. They need a target in a `derived-artifacts.yml` lane, a workflow edit those PRs could not make while draft PR #3036 changes the same file.
- `Tests/test_config_persistence_owner.py`, the whole file: 9 tests after TASK-35835, about 8 s. They pin that the quit leaves a fresh template byte-identical, upgrades v1 ciphertext in place, refuses a value written under another password between its reads, and writes nothing when there is nothing to do.
- `Tests/ProductionApp/test_config_profile_isolation.py::test_real_app_quit_persists_only_the_effective_config`. Run as a lone file, it errors at setup with `RecoveryRequired`; beside a sibling file that imports the app, it passes.
- `Tests/Wizards/test_first_run_setup_wizard.py::test_model_step_says_a_listed_pick_is_not_added_to_the_saved_model_list` (from TASK-35578).
- `Tests/Wizards/test_first_run_setup_integration.py::test_a_listed_model_setup_saves_leaves_the_saved_model_list_alone` (from TASK-35578).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each named test runs in a PR-required lane, with a matching Tests/CI lane-contract pin
- [x] #2 The real-quit test runs in that lane without the lone-file RecoveryRequired setup error
- [x] #3 The lane's measured job time stays at least 3 minutes under its cap
<!-- AC:END -->

## Implementation Plan

1. Lane: `pr-fast-lane-admission-2` (median 11.1 / max 13.6 min of its 35 over the last 28 successful PR runs; the other two PR Fast Lane jobs run 24.5/27.8 and 16.5/19.2). Its suites are bootstrap_profile, as the persistence file (Tests/conftest.py's `test_config_` rule) and the wizard file (module mark) are; the integration node is a private-profile child; the real-quit node runs its app in a subprocess with its own env.
2. Append the four targets at the tail of `Run admission-sensitive suites 2` and of `Tests/CI`'s `ADMISSION_2_TARGETS`, each with a rider note. Negative control: drop one target from the workflow, see the contract fail, restore.
3. AC#2: find why the lone file errors at setup; prove the real-quit node passes in the lane's exact invocation through run.sh (or fix its isolation if the lane would not run it beside an app-importing file).
4. AC#3: time the added targets in the step's order and flags, serial; run the whole step serially; estimate CI time with the ~2.6x ratio; record the CI job times read above.
5. Check the lane install line covers the tests' imports, including the two that spawn Python (a minimal venv with only the lane's install line).
6. Run Tests/CI with -n 4 and preflight.

## Implementation Notes

The four pins are appended to the tail of `pr-fast-lane-admission-2`'s `Run admission-sensitive suites 2` step and of `Tests/CI/test_ci_queue_pressure_contract.py`'s `ADMISSION_2_TARGETS`, with a rider note there. The workflow edit is those four target lines only (owner ruling of 2026-10-23 for this task).

- **Lane choice.** Over the last 28 successful PR runs (read 2026-10-23), job median / max: PR Fast Lane 24.5 / 27.8 min, admission-sensitive 16.5 / 19.2, admission-sensitive 2 11.1 / 13.6 (step median 612 s, max 759 s), all of 35. Admission-sensitive 2 has the most room. Its suites are bootstrap_profile, and so are the persistence file (Tests/conftest.py's `test_config_` rule) and the wizard node (the file's module mark), which cannot go in the sandboxed step. The integration node is a private-profile child.
- **AC#1 contract.** `test_second_admission_target_set_is_exact` pins the step exactly. Negative control: with the real-quit line dropped from the workflow it failed (1 failed, 83 passed); restored, it passes.
- **AC#2 cause.** `Tests/ProductionApp/conftest.py`'s autouse `_disable_model_catalog_refresh` calls `monkeypatch.setattr("tldw_chatbook.app.TldwCli._refresh_model_catalogs", ...)`. That string target imports `tldw_chatbook.app` at setup, under the per-test redirect, and the app's module-level `load_settings()` raises `RecoveryRequired('raw_source_selection_changed')`. The test file imports nothing from the app (its app runs in a subprocess), so as a lone file nothing has imported the app yet. In the step it passed only because 7 of its 23 files import the app at collection (the four new targets alone: 15 passed, 1 error, per the review).
- **AC#2 fix (review round 1).** The node is marked `@pytest.mark.bootstrap_profile`, as TASK-33641.4's definition-row subtitle node in the same tuple is, so it no longer depends on any neighbour. The kept profile cannot reach the app under test: its child's env is built from scratch, every profile root (HOME, USERPROFILE, XDG config/data/cache, TMPDIR, TLDW_CONFIG_PATH) under tmp_path, and from the parent only PATH, LANG, LC_ALL, TERM and COLORTERM. No guard edit (TASK-36381 owns the error class). Red/green through run.sh: the lone node unmarked gave 1 error (RecoveryRequired at setup); marked, 1 passed; the four targets alone, 16 passed.
- **AC#2 evidence.** The step's exact argv, extracted from the workflow (`pytest <64 targets> --timeout=300 --tb=short`, plus run.sh's `-p no:cacheprovider --basetemp` and a `--junitxml` for timings), serially through run.sh before the mark: 1038 passed in 1099 s, the real-quit node passing. Again in a scratch venv holding only the lane's install line (`pip install -e . pytest pytest-asyncio pytest-timeout packaging`, no hypothesis or xdist), so the two tests that spawn Python ran with the lane's imports: 1038 passed in 660 s.
- **AC#3, measured on PR #3222's CI (run 39993112363, head 0eb98f11fd).** `PR Fast Lane (admission-sensitive 2)` took 10.3 min wall, with its pytest step at 1038 passed in 566.83 s and all four added targets in the log. That is 24.7 min under the 35-min cap, against the prior median of 11.1 min and max of 13.6 min over 28 PR runs.
- **AC#3 (local estimate, before CI).** The 16 added cases took 39.9 s (main venv) and 48.7 s (lane-only venv) serially in the step, at load 30-43 on 18 CPUs. At the repo's ~2.6x CI/local ratio that is about 2.1 min. At this load's own ratio (CI step median 612 s against 660 s locally) it is under a minute. So expect a max of about 15.7 min, about 19 min under the 35-min cap. Even a hung test at `--timeout=300` (about 5.3 min with the private-profile report) ends well over 3 min under. The coordinator ticks this on the PR's measured run.
- Tests/CI with -n 4: 1025 passed, 1 skipped. Preflight: all checks passed.
- Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/ProductionApp/test_config_profile_isolation.py`.

