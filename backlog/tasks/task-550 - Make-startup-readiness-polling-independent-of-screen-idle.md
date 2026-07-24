---
id: TASK-550
title: Make startup readiness polling independent of screen idle
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 21:16'
updated_date: '2026-07-24 21:16'
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
- [ ] #1 The readiness helper yields asynchronously without invoking a global screen-idle barrier
- [ ] #2 The helper’s timeout remains enforceable when screen-idle would not complete
- [ ] #3 Startup performance tests pass in isolation and in the resumed suite
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
