---
id: TASK-16241
title: Bound Library entry synchronization waits
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 10:07'
updated_date: '2026-08-14 10:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the Library entry integration tests from hanging forever when background synchronization signals are never delivered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All eight architecture-reported waits are bounded.
- [x] #2 Library entry behavior tests remain green.
- [x] #3 The background-signal architecture guard and affected sweep chunk are green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Classify each reported event by whether the test owns the producer task.
2. Route task-owned signals through `wait_for_background_signal` and product-owned signals through `wait_for_signal`; bound completion of test-owned tasks as well.
3. Run the eight focused behavior tests, the background-signal guard, the full Library entry module, the sweep chunk, and static checks.

ADR required: no
ADR path: N/A
Reason: This is a test-synchronization correction using the repository's existing bounded-wait policy and helpers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Routed five task-owned start signals through `wait_for_background_signal` and their completions through `await_background_task`.
- Routed three product-owned worker signals through the timeout-only `wait_for_signal` helper.
- Verified 11 focused parametrized behaviors, the full Library-entry module plus architecture guard (86 passed), and the exact sweep chunk (336 passed, 23 optional/slow skips). Ruff check and diff hygiene pass; whole-file Ruff format remains pre-existing red outside the changed lines.
<!-- SECTION:NOTES:END -->
