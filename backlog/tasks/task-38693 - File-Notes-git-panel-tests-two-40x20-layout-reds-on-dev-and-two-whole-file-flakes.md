---
id: TASK-38693
title: "File Notes git panel tests: two 40x20 layout reds on dev and two whole-file flakes"
status: To Do
created_date: 2026-11-03 11:40
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
- [ ] #1 test_prepare_session_fixed_regions_remain_visible_at_40_by_20 and test_40x20_prepare_scrolls_actions_below_a_long_linked_root pass on dev, with the cause of each red recorded (product drift fixed, or the expectation updated with the reason)
- [ ] #2 test_commit_operation_observer_exception_does_not_invent_uncertain and test_commit_review_enter_escape_cancel_and_execution_are_state_aware pass 10 of 10 whole-file -n 4 runs
- [ ] #3 The whole file passes 3 of 3 -n 4 runs on dev
<!-- AC:END -->
