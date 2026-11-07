---
id: TASK-38694
title: "File Notes push tests: the same twelve nodes fail on every run on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 03:26
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_file_notes_git_push.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (reported as part of 24 local reds across the File Notes git, push and workspace test files). Checked on dev 9bbd5c2515, 2026-11-03: three whole-file runs of `Tests/UI/test_library_file_notes_git_push.py` (`-n 4`, load 18-28) each gave 12 failed and 52 passed, and the same 12 nodes failed every time. None of the failures is the local `RecoveryRequired` trap. They are:
- `test_workspace_local_proof_rechecks_candidate_after_trust_aba[blocked]` and `[ready]`, `test_workspace_preflight_review_rechecks_candidate_after_trust_aba`, `test_workspace_push_keyboard_happy_path_reaches_succeeded_result`, `test_endpoint_details_close_rejects_stale_push_operation_focus`: `assert False` on a Back button's `has_focus` (`#file-notes-git-back`, `#file-notes-git-push-back`, `#file-notes-git-push-back-session`);
- `test_workspace_push_review_admission_failure_has_safe_back`: "admission failure Back did not receive focus";
- `test_workspace_push_review_adopts_operations_and_restores_endpoint_details_focus_once`;
- `test_workspace_push_recovery_authorizes_then_queries_retained_endpoint`: "Back to Files did not hide the query-only recovery";
- `test_workspace_push_recovery_failure_replaces_stale_action[authorization]` and `[typed_admission]`;
- `test_workspace_rehydrate_hidden_settlement_without_duplicate_work[size0]` (`assert not True` on `#file-notes-editor-pane`'s display) and `[size1]` ("retained push did not open from Session Git").

The test file last changed on dev in 94df18af45 (2026-10-02, a formatter pass), so product code has moved under these tests. Most failures are about where focus lands after a push step closes. That is the area TASK-38647 found racing deferred focus (`call_later` or `call_after_refresh`) in the sibling workspace file. No PR lane runs this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the twelve nodes passes on dev, with the cause of each red recorded (a product regression fixed, or a stale expectation updated with the reason)
- [x] #2 The whole file passes 3 of 3 -n 4 runs on dev
<!-- AC:END -->

## Implementation Plan

1. Show the twelve nodes red at the base with whole-file runs, and probe where focus and display land for one of them.
2. Since 2026-08-29 Session Git (the entry button and the git panel) lives in the Manage region of the editor pane, and `_sync_navigator_mode` shows the panel only in Manage mode. These tests set only `_navigator_mode = "git"`, or press the entry from Edit mode, so the panel stayed hidden and focus could not land in it. Enter Manage first, through one helper that every Session Git setup uses.
3. The 40x20 rehydrate case expected Session Git in the narrow navigator view; it now opens in the editor view. Update those narrow-view expectations with the reason.
4. Run the whole file 3 times with -n 4.

## Implementation Notes

Branch fu/r6-fntests, base c2cdb6b82e. The work is a test-only change; no product regression was found. The Implementation Plan above was written after the first diagnosis, not before it.

- **Red at base:** a whole-file `-n 4` run gave 12 failed and 52 passed, and the 12 were exactly the nodes this task lists.
- **Cause, stale setup (nodes 1-11 below).** Since 7cf89de6c0 (2026-08-29), Session Git lives in the Manage region of the editor pane: the "Review session changes" entry and the git panel are both there. `_sync_navigator_mode` shows the panel only when `_work_mode == "manage"`. These tests last changed behaviour on 2026-08-20. Two things kept the panel hidden, so every focus wait on a control inside it failed:
  - most of them set `workspace._navigator_mode = "git"` in Edit mode;
  - the keyboard tests pressed the entry while Edit mode hid it.

  A probe in `test_workspace_local_proof_rechecks_candidate_after_trust_aba` showed this. After "Back to session" the push phase was "list", but `#file-notes-git-panel` was not displayed and focus sat on `#file-notes-root-details`.
- **Fix for those nodes.** A new helper, `_show_session_git`, enters Manage through the real button (`_show_manage`, from the git test file) and then shows the navigator. All 17 Session Git setups use it. The two keyboard journeys enter Manage before they focus the entry.
- Fixed by that change:
  1. `test_workspace_local_proof_rechecks_candidate_after_trust_aba[blocked]` and `[ready]`
  2. `test_workspace_preflight_review_rechecks_candidate_after_trust_aba`
  3. `test_endpoint_details_close_rejects_stale_push_operation_focus`
  4. `test_workspace_push_review_adopts_operations_and_restores_endpoint_details_focus_once`
  5. `test_workspace_push_review_admission_failure_has_safe_back`
  6. `test_workspace_push_recovery_authorizes_then_queries_retained_endpoint`
  7. `test_workspace_push_recovery_failure_replaces_stale_action[authorization]` and `[typed_admission]`
  8. `test_workspace_push_keyboard_happy_path_reaches_succeeded_result` (Manage before the keyboard journey)
  9. `test_workspace_rehydrate_hidden_settlement_without_duplicate_work[size1]` (120x40, the same)
- **`test_workspace_rehydrate_hidden_settlement_without_duplicate_work[size0]` (40x20): stale expectation.** The test went Back to the narrow navigator view and opened Session Git from there, expecting the navigator displayed and the editor pane hidden. Session Git is now in the editor pane, and Manage switches a narrow workspace to the editor view. The case now enters Manage and expects the editor view at three points: on entry, after the remount, and on the reopened result. At those points `_narrow_view == "editor"`, the editor pane is displayed, and the navigator is not. Its point is unchanged: leave and remount reuse the exact observer without starting Git again.
- **Green:** six whole-file `-n 4` runs, 64 passed each (load 15-28, 26-38 s).
- Files: `Tests/UI/test_library_file_notes_git_push.py`.
- 2026-11-04, review round 1: `test_push_focus_repair_and_buffered_enter_cannot_cross_operation` asserted that no message was posted at all. That held only because Enter lands before the review's deferred focus. It now asserts the safety property: no `push_reviewed_commit` message and no message for the stale operation 61. Red: with Push focused before Enter (a temporary edit replacing the review's deferred Back focus with a direct Push focus), it fails with `[PushOperationRequested()]`. Green on the branch.
