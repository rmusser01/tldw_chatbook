---
id: TASK-38693
title: "File Notes git panel tests: two 40x20 layout reds on dev and two whole-file flakes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 03:26
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI/test_library_file_notes_git.py
- tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
dependencies:
- TASK-36527
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36527's checks, PR #3377, which reported both 40x20 reds as red on dev). Checked on dev 9bbd5c2515, 2026-11-03, with three whole-file runs of `Tests/UI/test_library_file_notes_git.py` (`-n 4`, load 15-30) and node runs alone.

**Red on every run** (3 of 3 whole-file and 3 of 3 alone):
- `test_prepare_session_fixed_regions_remain_visible_at_40_by_20` (:2434) fails at its conflict row: `assert 'BLOCKED · .Conflict: use Git; Refresh' == 'BLOCKED · Conflict: use Git; Refresh'`, with a stray `.` before "Conflict".
- `test_40x20_prepare_scrolls_actions_below_a_long_linked_root` (:5096) fails with `AssertionError: linked-root summary did not settle to compact rendered text` (:5126).

The test file has not changed on dev since 2026-09-14, so product code has moved under both.

**Flaky in whole-file runs:**
- `test_commit_operation_observer_exception_does_not_invent_uncertain` (:6412) failed 2 of 3 whole-file runs with `assert None is not None`, and passed 12 of 12 runs alone. The report places the `None` at `_active_commit_key` (about :6427), at roughly 1 in 12 runs.
- `test_commit_review_enter_escape_cancel_and_execution_are_state_aware` (:1807) failed 1 of 3 with `assert 'form' == 'checking'`.

PR #3377 adds this file's new quit and removal tests to the UI census (8 entries, 65 cases). The rest of the file stays off every PR lane, so none of these reds reaches a check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_prepare_session_fixed_regions_remain_visible_at_40_by_20 and test_40x20_prepare_scrolls_actions_below_a_long_linked_root pass on dev, with the cause of each red recorded (product drift fixed, or the expectation updated with the reason)
- [x] #2 test_commit_operation_observer_exception_does_not_invent_uncertain and test_commit_review_enter_escape_cancel_and_execution_are_state_aware pass 10 of 10 whole-file -n 4 runs
- [x] #3 The whole file passes 3 of 3 -n 4 runs on dev
<!-- AC:END -->

## Implementation Plan

1. Conflict/unavailable row copy at 40x20: the row's content width is now 37 (a 1-cell list scrollbar), and `_row_secondary_copy` gives the 1-cell remainder to the elided reason, which renders "." glued to the standalone narrow recovery ("Conflict: use Git; Refresh"). Fix the fitter so a standalone narrow recovery replaces the reason; pin it at every narrow width with a pure test.
2. Observer flake: `_open_guarded_commit_form` returns when the panel shows the form, before the workspace adopts the draft from `CommitStagedRequested`, so `_active_commit_key()` can be None. Wait for the workspace's draft too.
3. Review Enter flake: the form focuses its subject after a refresh; the test focuses Review one pause after opening the form, so the deferred subject focus can take it back and Enter never reaches Review. Wait for the subject focus, then for Review's focus, before pressing Enter.
4. Show each red (flakes under a temporary delay of the raced step), then green; run the whole file 10 times with -n 4.

## Implementation Notes

- 2026-11-03 (branch fu/r5-filenotes2, TASK-36513/36528/36531): `test_40x20_prepare_scrolls_actions_below_a_long_linked_root` was red because of the folder row. The always-displayed "Review recovered pairing…" button left the root status 1 cell (".") at 40 columns, so the test's `"..."` wait never held. TASK-36513 now hides that button unless a pairing awaits review, and fits the row by parts, so the row reads "Linked · notes". The test's row expectation is updated to that form; the tooltip still carries the exact path. The rest of the test (Details disclosure, keyboard scrolling to the Git actions, stage/unstage) passed unchanged. The test passes alone and in whole-file `-n 4` runs on that branch. AC#1's other node, `test_prepare_session_fixed_regions_remain_visible_at_40_by_20`, still fails there with its stray "." before "Conflict"; this branch did not touch it.
- 2026-11-04 (branch fu/r6-fntests, base c2cdb6b82e). Load average 15-37 on 18 CPUs throughout.
  - **`test_prepare_session_fixed_regions_remain_visible_at_40_by_20`: product bug.** At 40x20 the Session Git row's content is 37 cells (40, less the list's 1-cell scrollbar and the row's 2-cell padding). TASK-1235 sized the narrow copy to exactly 36. `_row_secondary_copy` gave the 1 spare cell to the elided reason, which `_middle_elide_cells` renders as ".". That dot ran into the standalone narrow recovery: "BLOCKED · .Conflict: use Git; Refresh". The Git unavailable row became "BLOCKED · ; restore Git, then Refresh", an empty reason before its recovery. The fit now uses the narrow recovery once the full one leaves the reason fewer than 4 cells (one character beside "..."). A narrow recovery that does not start with ";" replaces the reason. The edit is net zero lines, so the module stays at its 4,259-line size-ratchet row.
  - New `test_blocked_row_copy_never_runs_an_empty_reason_into_its_recovery` checks both blocked states at every width from 36 to 119. Red before the fix: `(37, 'BLOCKED · .Conflict: use Git; Refresh')` and `(37, 'BLOCKED · ; restore Git, then Refresh')`. The 40x20 test was red at the same assertion and now passes.
  - **`test_commit_operation_observer_exception_does_not_invent_uncertain`: test race.** `_open_guarded_commit_form` waited for the panel's `commit_phase == "form"`. The panel sets that before the workspace adopts the draft from `CommitStagedRequested`, and the test reads `_active_commit_key()` next. The helper now also waits for that key. Negative control: a temporary 0.15 s sleep in the workspace's `_session_git_commit_staged` made the base test red at `assert key is not None`, the reported line. With the fixed helper, it and all 82 `-k commit` tests passed under that sleep. It also failed in the first base whole-file run.
  - **`test_commit_review_enter_escape_cancel_and_execution_are_state_aware`: test race.** The form focuses its subject through a deferred `call_after_refresh`. The test focused Review one pause after opening the form, so the subject focus sometimes landed after it and took Enter. Reproduced on the base test, 1 of 60 repeats under `-n 4`: focus was on the list surface before Enter and on the subject after it. The test now waits for the subject's focus after each form open, then for Review's focus, before pressing a key. 180 of 180 repeats passed.
  - **A third flake from the AC#3 runs:** `test_guarded_commit_success_renders_and_focuses_fresh_owner_status[size1]` (40x20) was red in 1 of 10 whole-file runs. Base: 5 of 48 repeats red. A diagnostic repeat showed that in every run the summary settles at 27 cells, fitted and middle-elided. The list surface's scrollbar takes a cell after the first fit at 28, and the panel does not resize. The test's `startswith("Committed 2 session notes")` held only on the stale 28-cell fit. It now waits until the summary equals `_fit_two_line_copy` at its current width, then checks the elided form's head and tail. 120 of 120 repeats passed (60 at each size). No product change: the copy does settle to its real width. A lesson records this.
  - **Whole file, `-n 4`:** 10 runs before that third fix, with 9 green; the one red was the third flake. AC#2's two tests passed all 10. After it, 5 of 5 green (223 passed, 64-72 s each).
  - Docs: `Docs/User_Guide/library/file-notes.md` gives the narrow blocked-row copy. Lesson added to `backlog/docs/lessons-testing-evidence.md`.
  - Files: `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`, `Tests/UI/test_library_file_notes_git.py`.
- 2026-11-04, review round 1 (fu/r6-fntests):
  - **Narrow rows.** Other states still ran an empty or dot-only reason into their "; " narrow recovery: ignored at widths 36-39 ("BLOCKED · ; fix ignore, then Refresh", then ".", "..", "..."); the externally staged and unsupported-family states at 33-36; failed at 25-28. When the narrow recovery still leaves the reason under 4 cells, the reason and the leading "; " are now dropped, as in "BLOCKED · fix ignore, then Refresh". The module stays at 4,259 lines.
  - The pure test is renamed `test_row_copy_never_runs_an_empty_or_dot_only_reason_into_its_recovery`, and its census entry with it. It covers all ten states that carry a reason, at every width from where the short recovery fits up to 119. The check: the state word is followed by a letter, never "." or ";". Red before the fix for the eight "; " states (for example `(34, 'BLOCKED · ; fix ...e, then Refresh')`); green after. The User Guide sentence now gives the ignored example too.
  - **Comment and lesson.** The commit-success wait's comment and the lesson now say: wait until the copy matches its fit for the current width (28 or 27 cells at 40x20, both middle-elided).
  - **`test_commit_footer_keeps_disclosure_edit_cancel_confirm_order_and_geometry[size0]`.** The reviewer saw it fail 1 of 3 at load 34, at `assert edit.has_focus` one pause after the review rendered. The review focuses Edit through `call_after_refresh`, so this is a test race; the test now waits on that focus. Red: with that focus delayed 0.3 s (temporary edit), the old test failed at the assertion and the new one passed. The failure itself did not reproduce here: 0 of 480 [size0] repeats at load 10-23, and 0 of 10 whole-file runs run five at a time at load 30-35.
  - **Other one-off failures under heavy load.** One five-at-a-time set of whole-file runs, at load 34 with the 80 repeats added, had five one-off failures in tests this branch does not touch:
    - `test_row_action_table_is_driven_by_row_policy[owned_topology_changed-stage_update-False]`;
    - `test_session_git_actions_reveal_only_the_current_next_steps[size1]`;
    - `test_unstage_selected_reports_counts_and_refreshes_once`;
    - `test_stage_rechecks_transition_admission_after_flush_await`;
    - `test_opening_session_git_moves_focus_to_a_visible_ready_control`.

    The plain five-at-a-time rerun at head was 5 of 5 green. The same command at base also had a one-off failure: `test_mutation_blocks_root_and_path_while_path_lease_blocks_mutation`.
  - **`test_stage_draft_conflict_names_save_and_editor_recovery`.** The review round's own whole-file run, at load 25-32, failed it once: the status still read "READY" one pause after the Stage press. The press reaches the workspace's `StageRequested` handler through two messages, so the test now waits for the refusal's status text. Red: a temporary 0.2 s sleep in that workspace handler reproduced the exact failure on the old test; the new test passed under it.
  - After round 1, `-n 4` whole-file runs at load 13-27: git 3 of 3 (231 passed); push 3 of 3 (64 passed); workspace 2 runs with only TASK-32185's six excluded reds failing (166 passed).
