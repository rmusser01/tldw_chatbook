---
id: TASK-41530
title: "test_finishing_card_is_not_counted_and_keyboard_focuses_the_card flakes at load and sits in a whole-file census entry"
status: To Do
assignee: []
created_date: 2026-11-17 05:39
labels:
- follow-up
- tests
- flaky
- console
priority: medium
references:
- Tests/UI/test_console_mcp_approval.py
dependencies:
- TASK-41138
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by testinfra (round 15). `test_finishing_card_is_not_counted_and_keyboard_focuses_the_card` (`Tests/UI/test_console_mcp_approval.py:2167`) fails at `-n 4`, load about 90, because `screen._console_attach_reconciled` is still False when the test reads it (asserts at about `:1640`/`:1655`). The file is a whole-file entry in `scripts/ui_pr_gate_census.txt`, so the flake can redden the required check. TASK-41138 only listed it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test waits on `_console_attach_reconciled` (a condition wait with a bounded deadline) and passes 10 of 10 at `-n 4`, load recorded
- [ ] #2 The same pattern is grepped across the file and each other fixed read-after-event is listed or fixed
<!-- AC:END -->
