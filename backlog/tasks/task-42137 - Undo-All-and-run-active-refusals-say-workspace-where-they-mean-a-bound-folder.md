---
id: TASK-42137
title: "Undo All and run-active refusals say workspace where they mean a bound folder"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 19:05
labels:
- follow-up
- console
- change-review
- copy
priority: low
references:
- tldw_chatbook/UI/Console_Modules/review_selection.py
- tldw_chatbook/UI/Screens/change_review_screen.py
- Tests/UI/test_change_review_screen.py
- Docs/User_Guide/console/agent-runs-and-tools.md
- tldw_chatbook/Workspaces/change_revert.py
- tldw_chatbook/Workspaces/change_revert_ownership.py
- tldw_chatbook/Workspaces/git_workspace.py
- Tests/UI/test_change_review_commit_ui.py
- Tests/UI/test_change_review_push_ui.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the docs stream's undo section (TASK-42003, round 18). Undo All's refusals name a change snapshot's `root` folder as a "workspace":
- `tldw_chatbook/UI/Console_Modules/review_selection.py:395`: "Undo All is unavailable because a change snapshot has no workspace.";
- `review_selection.py:400`: "This turn has multiple change windows for the same workspace." (the same text at `UI/Screens/change_review_screen.py:167`, pinned by `Tests/UI/test_change_review_screen.py:1323`);
- `review_selection.py:406` and `:466`: "A run is active on this workspace — finish or stop the run first."

The same run-active refusal appears on the Review screen's other paths, so after a change to the Console notices alone the Review screen would say "folder" in one place and "workspace" in another:
- `tldw_chatbook/Workspaces/change_revert.py:185`, `Workspaces/change_revert_ownership.py:382` and `Workspaces/git_workspace.py:924`: "a run is active on this workspace — finish or stop the run first";
- `UI/Screens/change_review_screen.py:100`, `COMMIT_RUN_ACTIVE_REFUSAL`. Tests use it by name (`Tests/UI/test_change_review_commit_ui.py:507`, `Tests/UI/test_change_review_push_ui.py:467` and `:474`), so they follow a wording change. (`test_change_review_commit_ui.py:853` asserts its own fake's text, not the product's.)

The checks are per root folder (`row.get("root")`), and a workspace can bind several folders. The undo section TASK-42003 restored (round-18 docs branch) says "folder".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each Undo All refusal that refers to a change snapshot's root names it as a folder, consistent with the User Guide's undo section
- [x] #2 The run-active refusals in `change_revert.py`, `change_revert_ownership.py`, `git_workspace.py` and `COMMIT_RUN_ACTIVE_REFUSAL` use the same wording, so Console and the Review screen name the same thing the same way
- [x] #3 Tests that pin the old wording are updated, and a test pins each new notice
- [x] #4 Where `Docs/User_Guide/console/agent-runs-and-tools.md` quotes these notices, it quotes them as they now read, including the commit refusal quoted verbatim at lines 1229-1230
- [x] #5 (2026-11-21, coordinator) The files that pin this copy run on a PR lane, with times recorded
<!-- AC:END -->

## Implementation Plan

1. Pin the new wording in tests first and show them red: tighten the partial matches in `Tests/UI/test_console_turn_undo_all.py`, `Tests/Workspaces/test_change_revert.py`, `Tests/Workspaces/test_git_workspace_commit.py` and `Tests/UI/test_change_review_screen.py`; add pins for the Console no-folder refusal, the Console apply-time run-active refusal, the native undo revalidation refusal and the `COMMIT_RUN_ACTIVE_REFUSAL` literal.
2. Change the copy: the engine and Review screen run-active refusals read "a run is active in this folder — finish or stop the run first"; Console's Undo All refusals say "folder" (no folder, multiple windows for the same folder, a run active in a folder this turn changed); `MULTI_WINDOW_UNDO_REFUSAL` says "folder".
3. Update the User Guide's quote of the commit refusal.
4. Run the touched test files at -n 4, Tests/Docs, preflight, the ruff gates and the markup census check.

## Implementation Notes

Every refusal that meant a snapshot's root folder said "workspace". It now says "folder"; the copy change is all this task does.

**New wording**
- Engine and Review screen, one sentence at all four sites: "a run is active in this folder — finish or stop the run first". The sites are `Workspaces/change_revert.py:185`, `Workspaces/change_revert_ownership.py:382`, `Workspaces/git_workspace.py:924` and `COMMIT_RUN_ACTIVE_REFUSAL` (`UI/Screens/change_review_screen.py:100`). Each check is per root (`run_active_for_root`, or the commit's repository root).
- Console Undo All (`UI/Console_Modules/review_selection.py`):
  - "Undo All is unavailable because a change snapshot has no folder."
  - "This turn has multiple change windows for the same folder." (the same sentence starts `MULTI_WINDOW_UNDO_REFUSAL`)
  - "A run is active in a folder this turn changed — finish or stop the run first." at :406 and :466. These two checks cover every root of the turn (`_console_turn_undo_run_active`), so "this folder" would not say which folder on a multi-folder card. The sentence matches the guide's "refuses while a run is active in any of the turn's folders". Both surfaces now name the thing "folder".
- Docstrings (coordinator decision 2026-11-21): `Workspaces/change_review_provider.py` `revert` and `commit_selected` now say the run is active in the row's root folder and in the repository's folder, so code and copy agree.
- User Guide: the commit refusal quote in "Commit and push from the working tree" now reads as the product does, and its lead-in says "active in that repository's folder". No other page quotes these notices.

**Tests**
- Pins tightened from partial `match=` to exact text:
  - `test_console_turn_undo_all.py`: the duplicate-root test and both run-active tests;
  - `test_change_revert.py::test_revert_refuses_while_a_run_is_active`;
  - `test_git_workspace_commit.py::test_run_active_refuses_before_touching`;
  - `test_change_review_screen.py::test_undo_all_refuses_same_root_windows_before_preflight` (literal updated).
- New pins:
  - `test_console_turn_undo_all.py::test_undo_all_refusals_name_a_missing_root_and_a_late_run_as_folders` covers the no-folder refusal and the apply-time run-active refusal.
  - `test_native_confirmed_undo.py::test_recheck_after_confirmation_names_the_active_run_folder` covers `_UndoCall.revalidate`, which had no run-active test.
  - `test_change_review_commit_ui.py::test_active_run_refuses_before_any_modal_or_status_read` now pins the `COMMIT_RUN_ACTIVE_REFUSAL` literal.
- The fake refusal in `test_an_unexpected_commit_error_is_not_dressed_as_a_refusal` now raises `COMMIT_RUN_ACTIVE_REFUSAL` instead of the old literal.
- Red/green:
  - With the base source, 9 pins failed on assertions (all of the above; the new Console test's first assertion: 'has no workspace.' != 'has no folder.'). All were green after the change.
  - The new test's second half was not reached in that red run, so a negative control checked it: line 466 reverted to "on this workspace" made it fail on the apply assertion. Reverted afterwards.
- None of the seven touched test files ran in a PR lane. AC #5 (added by the coordinator) lanes them; see "Laning" below.

**Runs** (through the wrapper, load average about 14 on 18 CPUs)
- The seven touched files at -n 4: 199 passed and 3 failed.
- The 3 failures are the `test_mounted_*` Console tests, which fail locally with `RecoveryRequired: raw_source_selection_changed` (the known trap). Under a temporary `@private_profile_test` marker (since removed), two passed. `test_mounted_partial_and_provider_failures_name_problem_and_allow_retry` failed with `NoMatches: '#change-revert-yes'`, and fails the same way with the base source restored (2 of 2). It is pre-existing and does not touch this copy.
- Module and screen size ratchets: 234 passed (serial).
- `ruff --select F`: clean. `ruff_gate.py`: rc 0.
- `check_markup_interpolation.py`: OK. The changed notices are static text.

**Laning (AC #5, coordinator decision 2026-11-21)**
- UI census tail (`scripts/ui_pr_gate_census.txt`, `MINIMUM_FILES` 2441 -> 2445); no re-deal. Each was run serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`) through the wrapper:

  | File | Tests | Time | Load |
  | --- | --- | --- | --- |
  | `Tests/UI/test_console_turn_undo_all.py` | 16 passed | 47.7, 55.8, 52.1 s (3 runs) | 40-44 |
  | `Tests/UI/test_change_review_screen.py` | 61 passed | 273.9 s | 45-52 |
  | `Tests/UI/test_change_review_commit_ui.py` | 27 passed | 68.8 s | 48-52 |
  | `Tests/UI/test_change_review_push_ui.py` | 33 passed | 70.9 s | 45-47 |
- `derived-artifacts.yml`, with `Tests/CI/test_ci_queue_pressure_contract.py` pins updated:
  - `Tests/Workspaces/test_change_revert.py` (11 in 31.3 s) and `test_git_workspace_commit.py` (20 in 9.6 s) join the sandboxed "Run fast PR contract" step (`FAST_LANE_TARGETS`). Both were measured with that step's flags at load 46-49.
  - `Tests/Workspaces/test_native_confirmed_undo.py` is a `bootstrap_profile` module, so it joins pr-fast-lane-admission (`ADMISSION_TARGETS`): 34 in 195.8 s at load 48, `--timeout=300`. That job ran 15.5-19.2 of its 35 min over six recent PR runs (2026-11-20/21), the most headroom of the admission jobs. The sandboxed job ran 27.1-29.7 min, so it took only the two small files.
- Profile setup:
  - `test_console_turn_undo_all.py`'s three mounted nodes failed locally with `RecoveryRequired: raw_source_selection_changed`. They now carry `@pytest.mark.bootstrap_profile`, as 512+ census entries do (TASK-36478).
  - That exposed `test_mounted_partial_and_provider_failures_name_problem_and_allow_retry`'s click before `#change-revert-yes` existed: NoMatches in 2 of 2 runs with the base source and 1 of 1 with the marker, so it blocked laning the file.
  - Fixed in the TASK-41505 shape: `_click_modal_button` waits until the confirmation has laid the button out, then asserts the click landed. All four confirmation clicks in the file use it. The file then ran 16 passed in 3 of 3 serial runs.
  - The other six files were green as they were.
- Not verified: a completed Linux CI run for the newly laned files (local macOS only).
- Checks after laning (wrapper, load 46-53): `Tests/CI` at -n 4 1811 passed, 2 skipped; preflight passed (census 2445, floor 2445); `ruff --select F` clean; `ruff_gate.py` rc 0; module and screen size ratchets 234 passed.
