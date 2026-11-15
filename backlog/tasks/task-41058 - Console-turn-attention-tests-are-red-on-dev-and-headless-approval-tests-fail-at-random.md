---
id: TASK-41058
title: "Console turn-attention tests are red on dev, and headless-approval tests fail at random"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
priority: low
references:
- Tests/UI/test_console_turn_attention.py
- Tests/UI/test_console_headless_approval.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15), under the bootstrap profile at `-n 4`, two Console attention test files fail:
- **`Tests/UI/test_console_turn_attention.py`** fails three tests every run, with "'types.SimpleNamespace' object has no attribute '_current_console_conversation_id'": `test_successful_full_row_refresh_acknowledges_after_exact_mount`, `test_outgoing_refresh_cannot_ack_after_successor_claims_runtime` and `test_failed_full_row_refresh_acknowledges_nothing`. A code path reads `_current_console_conversation_id` from the harness's `SimpleNamespace` stand-in, which does not define it. TASK-36947.5's review saw the same failures on 2026-11-06.
- **`Tests/UI/test_console_headless_approval.py`** failed five tests in this run, among them `test_two_headless_rounds_each_mount_in_turn` and `test_a_wake_delivery_cannot_resolve_a_pending_headless_round`. The failing set changes from run to run on both sides of every Phase 3 comparison, as TASK-36947.4's review rounds and TASK-34100.5 recorded.

A suite that is always partly red cannot show which approval regression is new. Neither file runs on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three turn-attention tests pass on dev; the task notes say whether the harness or the product changed
- [ ] #2 test_console_headless_approval.py passes 3 of 3 runs at `-n 4`; each test that failed has a named cause in the task notes
<!-- AC:END -->
