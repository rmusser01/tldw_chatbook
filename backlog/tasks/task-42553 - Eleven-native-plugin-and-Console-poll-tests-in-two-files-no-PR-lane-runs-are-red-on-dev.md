---
id: TASK-42553
title: "Eleven native-plugin and Console poll tests in two files no PR lane runs are red on dev"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-26 04:50
updated_date: '2026-11-26 05:18'
labels:
- follow-up
- console
- plugins
- dev-red
priority: medium
references:
- Tests/Chat/test_native_plugin_hook_preparation.py
- Tests/UI/test_console_poll_reconciliation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 26 (TASK-42551's caller runs, 2026-11-25). Run serially on dev 692acb43dc, these 11 tests fail; no PR lane runs their files, so nothing reports them. If the product is wrong, a send can be refused with "Required hook initialization failed.", a plugin admission can run outside native activity, or a poll can fail to settle.

- `Tests/Chat/test_native_plugin_hook_preparation.py` (8): `test_activity_busy_is_original_visible_submission_refusal`, `test_actual_plugin_admission_has_one_native_operation`, `test_cancelled_preparation_keeps_original_admission_until_return`, `test_first_use_original_resource_replacement_refuses_before_plugin_worker[activity]` and `[capacity]`, `test_only_original_manual_submission_can_use_reserved_capacity[True]`, `test_original_host_check_cannot_accept_replaced_native_source`, `test_original_native_result_handoff_remains_owned`. Errors include "Required hook initialization failed." and "test failed to release actual plugin admission".
- `Tests/UI/test_console_poll_reconciliation.py` (3): `test_full_request_during_poll_await_replays_with_current_owner[original-demand]` and `[preparing-display]`, `test_poll_terminal_transition_settles_while_background_turn_remains_active`.

The seven red nodes of `Tests/Chat/test_native_v2_mcp_events.py` in the same run, and its teardown timeout, are TASK-41061. Three red tests in `Tests/UI/test_console_staged_evidence_sessions.py` are left to open PR #3707 (TASK-33620.7), which repairs that file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each red test, the notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [ ] #2 A product defect is fixed and its test shown red before and green after; a drifted test is repaired with every behaviour assertion kept
- [ ] #3 Both files pass on dev at `-n 4`, twice in a row, and each pytest process exits on its own
- [ ] #4 Both files run on a PR lane that blocks merges to dev, timed with that lane's flags
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run both files at base 6e41919880, at the #3480 merge (e10ce63fe8) and at the dev commits around each first red, to name each node's cause and the commit that introduced it (scratch probes only, no product edits).
2. Decide per node whether the product or the test is wrong. Fix a product defect with its test red before and green after; repair a drifted test with every behaviour assertion kept.
3. Run both files at `-n 4` twice in a row and serially, and check each pytest process exits on its own.
4. Lane the poll file at the UI census tail (set `MINIMUM_FILES`) and the hook file in the recovered-suites step, timed with each step's flags, and update the lane contract tests.
<!-- SECTION:PLAN:END -->
