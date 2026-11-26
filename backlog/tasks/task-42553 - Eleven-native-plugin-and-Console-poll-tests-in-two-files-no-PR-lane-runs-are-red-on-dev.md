---
id: TASK-42553
title: "Fourteen native-plugin and Console UI tests in three files no PR lane runs are red on dev"
status: To Do
assignee: []
created_date: 2026-11-26 04:50
labels:
- follow-up
- console
- plugins
- dev-red
priority: medium
references:
- Tests/Chat/test_native_plugin_hook_preparation.py
- Tests/UI/test_console_poll_reconciliation.py
- Tests/UI/test_console_staged_evidence_sessions.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 26 (TASK-42551's caller runs, 2026-11-25). Run serially on dev 692acb43dc, these 14 tests fail; no PR lane runs their files, so nothing reports them. If the product is wrong, a send can be refused with "Required hook initialization failed.", a plugin admission can run outside native activity, a poll can fail to settle, or staged evidence can cross chats.

- `Tests/Chat/test_native_plugin_hook_preparation.py` (8): `test_activity_busy_is_original_visible_submission_refusal`, `test_actual_plugin_admission_has_one_native_operation`, `test_cancelled_preparation_keeps_original_admission_until_return`, `test_first_use_original_resource_replacement_refuses_before_plugin_worker[activity]` and `[capacity]`, `test_only_original_manual_submission_can_use_reserved_capacity[True]`, `test_original_host_check_cannot_accept_replaced_native_source`, `test_original_native_result_handoff_remains_owned`. Errors include "Required hook initialization failed." and "test failed to release actual plugin admission".
- `Tests/UI/test_console_poll_reconciliation.py` (3): `test_full_request_during_poll_await_replays_with_current_owner[original-demand]` and `[preparing-display]`, `test_poll_terminal_transition_settles_while_background_turn_remains_active`.
- `Tests/UI/test_console_staged_evidence_sessions.py` (3): `test_a_send_in_one_chat_leaves_another_chats_staging_alone`, `test_original_stock_handler_accepts_b_without_consuming_a_staging`, `test_staged_evidence_follows_one_turn_and_one_chat`.

The seven red nodes of `Tests/Chat/test_native_v2_mcp_events.py` in the same run, and its teardown timeout, are TASK-41061.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each red test, the notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [ ] #2 A product defect is fixed and its test shown red before and green after; a drifted test is repaired with every behaviour assertion kept
- [ ] #3 All three files pass on dev at `-n 4`, twice in a row, and each pytest process exits on its own
- [ ] #4 The three files run on a PR lane that blocks merges to dev, timed with that lane's flags
<!-- AC:END -->
