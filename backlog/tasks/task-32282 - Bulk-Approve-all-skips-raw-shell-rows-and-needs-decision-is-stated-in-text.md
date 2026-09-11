---
id: TASK-32282
title: Bulk Approve all skips raw-shell rows and needs-decision is stated in text
status: Done
assignee: []
created_date: '2026-09-10 19:13'
updated_date: '2026-09-10 20:44'
labels:
  - console
  - approvals
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Approve all' moves a raw-shell row off its deliberate Deny default, and the needs-decision text prefix is never produced (no producer sets it; the bulk-skip path only adds a CSS class), so the state is colour-only. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approve all leaves raw-shell rows on Deny and flags them as needing an explicit decision.
- [x] #2 A row skipped by a bulk action shows a 'needs decision' prefix in its header text until it is decided.
- [x] #3 Tests cover both behaviours.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add two index-parallel lists to `ChatApprovalCard` (alongside `_batch_rows`): `_batch_is_raw_shell` (bool per row) and `_batch_base_headers` (the row's plain header text, pre-prefix), populated everywhere `_batch_rows` is populated (`set_batch`'s row-build loop, `_update_mounted_single_row`, and the empty-calls reset).
2. Add `_mark_row_needs_decision`/`_clear_row_needs_decision` helpers that toggle the `needs-decision` CSS class AND the header Static's text (`NEEDS_DECISION_PREFIX + base_header` / `base_header`).
3. In `_set_all_batch_decisions`, skip the legality search entirely for a raw-shell row when `"approve_once" in candidates` (Approve all), so its Select is left on Deny; route every row's applied/unapplied outcome through the mark/clear helpers instead of touching the CSS class directly.
4. In `_on_batch_row_select_changed`, clear via the same helper so a user's own explicit choice removes the text prefix along with the class.
5. Add failing-first widget-mounted tests for the three brief cases, run red, implement, run green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed both bugs in ChatApprovalCard (tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py): Approve all no longer moves a raw-shell row off its Deny default, and a row a bulk action skips now carries NEEDS_DECISION_PREFIX on its header Static text, not just the needs-decision CSS class. Two new index-parallel lists (_batch_is_raw_shell, _batch_base_headers) track per row whether it is the raw-shell row and its plain header text, populated at every site that populates _batch_rows; _set_all_batch_decisions skips the legality search for a raw-shell row when approve_once is a bulk candidate (Approve all), and every row's applied/unapplied outcome now routes through new _mark_row_needs_decision/_clear_row_needs_decision helpers (toggling class and text together); _on_batch_row_select_changed clears via the same helper on the user's own explicit choice. Deviated from the brief's named test files (test_chat_approval_card.py is pure-function only, test_console_raw_shell_approval.py covers the review hook not the widget) and added 3 new tests to Tests/UI/test_console_mcp_approval.py instead, which already owns the mounted-card bulk-approve/deny harness and the _raw_shell_call fixture; confirmed RED before the fix, GREEN after. Baseline and post-fix runs of the 3 relevant test files show the same 4 pre-existing, unrelated failures (123->126 passed); a broader sweep of 10 other ChatApprovalCard-referencing test files found 14 failures, all in test_console_workbench_contract.py and confirmed pre-existing (unrelated header/geometry assertions) by reproducing 2 of them with this diff stashed out.
<!-- SECTION:NOTES:END -->
