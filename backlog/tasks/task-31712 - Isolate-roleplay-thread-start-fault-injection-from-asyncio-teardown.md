---
id: TASK-31712
title: Isolate roleplay thread-start fault injection from asyncio teardown
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 18:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve thread startup failure coverage without replacing the standard library Thread class used by the test runner during cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Constructor and start failures still release the fork transition without changing canonical state
- [ ] #2 The standard library Thread identity remains unchanged during fault injection and pytest teardown emits no runner warning
- [ ] #3 The full Console session settings file passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only isolation of an existing dependency fault injection.
1. Reproduce the asyncio teardown warning and inspect the screen threading calls.
2. Patch only the screen module threading namespace and assert the stdlib identity remains unchanged.
3. Run the focused variants with warnings escalated and the complete settings file; run static checks.
<!-- SECTION:PLAN:END -->
