---
id: TASK-38694
title: "File Notes push tests: the same twelve nodes fail on every run on dev"
status: To Do
created_date: 2026-11-03 11:40
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
- [ ] #1 Each of the twelve nodes passes on dev, with the cause of each red recorded (a product regression fixed, or a stale expectation updated with the reason)
- [ ] #2 The whole file passes 3 of 3 -n 4 runs on dev
<!-- AC:END -->
