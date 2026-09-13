---
id: TASK-22527
title: Repair missed Console durable-send test harnesses
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 04:29'
updated_date: '2026-08-26 04:34'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the durable-send precondition in Console send-state integration tests so they exercise the accepted-send and queued-draft behavior instead of failing closed before dispatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Enter hotkey dispatch test mounts with a durable in-memory conversation database and reaches the configured gateway.
- [x] #2 The queued-draft test mounts with the same durable persistence precondition and verifies the accepted-run queue behavior.
- [x] #3 The full Console send-disabled-state test module passes without changing production send behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing failing integration tests as RED evidence and confirm the durable database attachment is the differentiating precondition.
2. Attach the standard in-memory conversation database in only the two real-send harnesses that currently omit it.
3. Run the two repaired tests, the complete send-disabled-state module, formatting/static checks, and a focused regression set.
4. Record implementation evidence and complete task hygiene.

ADR required: no
ADR path: N/A
Reason: This is a test-harness correction that restores an established durable-send precondition without changing production architecture or behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Attached the standard `attach_chachanotes_db()` in-memory durable conversation store before mounting the two real-send harnesses that had been missed by TASK-21590. The tests now reach their configured gateways and queue registry instead of failing closed at persistence; production code and runtime behavior are unchanged. Ruff also normalized the touched test module. ADR required: no; ADR path: N/A; Reason: test-only restoration of an established persistence precondition. TDD evidence: both focused tests failed before the repair with the database-unavailable transcript path, then passed after the two fixture attachments. Verification: focused pair 2 passed; complete `test_console_send_disabled_state.py` module 9 passed; Ruff lint and format checks passed; `git diff --check` passed. The broader adjacent suite exposed a separate Anthropic provider-readiness harness failure in `test_console_cost_chip_screen.py`; that is being tracked independently rather than conflated with this task.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

- Former ID: `TASK-22302`.
- Renumbered to `TASK-22527` on 2026-08-26 after rebasing onto `dev`, where the
  older-arriving citation-provenance task already owned `TASK-22302`.
- The dependent Console cost-chip harness task now references `TASK-22527`.
