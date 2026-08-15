---
id: TASK-16206
title: Repair prompt-queue context-change fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 06:45'
updated_date: '2026-08-14 06:45'
labels:
  - test-health
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore prompt-queue context-drift evidence without editing an active ancestor, which now intentionally invalidates the in-flight assistant and tests stop behavior instead of queue admission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test mutates conversation context without terminating the active turn.
- [x] #2 Queued work pauses as CONTEXT_CHANGED and is not dispatched.
- [x] #3 Focused/coordinator and containing-chunk gates plus scoped static/diff evidence pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: test setup is aligned with existing context-epoch and ancestor-edit contracts.

1. Reproduce the STOPPED outcome and verify active-ancestor edits intentionally invalidate descendants.
2. Use the existing session-summary mutation to advance the context epoch without disturbing the active response.
3. Prove the old edit shape fails, then run the focused coordinator, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed the context-drift fixture from an active-ancestor content edit to the store's existing session-summary mutation. Both advance the conversation context epoch, but only the former intentionally invalidates the in-flight assistant and therefore exercises STOPPED rather than queue admission. RED: the original isolated node consistently paused as STOPPED. GREEN: all 24 coordinator tests and the 25-file containing chunk (814 tests, including the localhost-loop gateway tests) passed. Scoped Ruff check and diff-check passed; the touched test file is Ruff-format-red identically on the implementation base, so no unrelated formatting churn was introduced. ADR required: no; test setup only.
<!-- SECTION:NOTES:END -->
