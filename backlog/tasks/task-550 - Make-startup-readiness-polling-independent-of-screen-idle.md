---
id: TASK-550
title: Make startup readiness polling independent of screen idle
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 21:16'
updated_date: '2026-07-26 07:45'
labels:
  - tests
  - performance
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the startup readiness guard bounded under full-suite load by polling state without waiting for Textual’s entire screen message queue to become idle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The readiness helper yields asynchronously without invoking a global screen-idle barrier
- [x] #2 The helper’s timeout remains enforceable when screen-idle would not complete
- [x] #3 Startup performance tests pass in isolation and in the resumed suite
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a deterministic helper regression whose external deadline fails if polling waits on an indefinitely blocked pause callback.
2. Replace screen-idle polling with asyncio.sleep and simplify readiness call sites.
3. Run the focused startup-performance file, static checks, and resume after Performance.

ADR required: no
ADR path: N/A
Reason: This is test-harness stabilization that preserves the existing startup lifecycle and performance contract without changing application behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Made startup readiness polling bounded independently of Textual screen-idle state.

Implementation:
- Replaced the injected Pilot.pause polling callback with asyncio.sleep on each bounded interval.
- Kept the event-loop monotonic deadline and final condition check.
- Added a regression wrapped in asyncio.wait_for that proves readiness can advance and complete without a screen-idle callback.
- Simplified startup call sites to use the state-polling helper directly.

Verification:
- Focused startup performance module: 7 passed.
- Diagnostic/task sentinel harness: 2 passed.
- Final permitted full suite: 12,757 passed, 231 skipped, 240 warnings in 3h34m55s.
- Self-review: test-harness-only stabilization; production startup behavior is unchanged.

ADR required: no
ADR path: N/A
Reason: Test-harness stabilization preserving the existing startup lifecycle and performance contract.

Files modified:
- Tests/Performance/test_app_startup_performance.py
- backlog/tasks/task-550 - Make-startup-readiness-polling-independent-of-screen-idle.md
<!-- SECTION:NOTES:END -->
