---
id: TASK-31557
title: Restore roleplay writer transition fallback cleanup
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:28'
updated_date: '2026-09-05 01:30'
labels:
  - console
  - tests
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent a completed app-owned Console roleplay writer from leaking its fork-source transition when its screen waiter is cancelled before accepting the persistence result.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Writer completion schedules fallback abandonment after owner acceptance has had a turn.
- [x] #2 Unmounted or cancelled screen waiters cannot leak a fork-source transition.
- [x] #3 The focused cleanup and startup-failure regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the stale helper failure and compare the pre-owner callback contract with the current app-owned durability task lifecycle.
2. Restore the minimal idempotent fallback callback and register it after completion consumption so ordinary owner acceptance remains authoritative.
3. Run the cleanup regression, cancellation/startup lifecycle coverage, Ruff, and diff checks.

ADR required: no
ADR path: N/A
Reason: this restores a lifecycle safety invariant within the established Console store and app-owned durability boundary; it introduces no new boundary or policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Restored the idempotent writer-completion fallback removed during the app-owned durability migration and registered it after completion consumption. The fallback is queued one event-loop turn later, so normal result acceptance releases the token first; cancellation/unmount still cannot strand it.
- Evidence: five focused writer cleanup, startup-failure, cancellation, and serialization lifecycle cases pass.
- ADR required: no; this restores the existing fork-safety invariant within the current app-owned task boundary.
<!-- SECTION:NOTES:END -->
