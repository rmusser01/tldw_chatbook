---
id: TASK-19507
title: Make Console view-hook binding tolerate partial screen initialization
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21 18:04'
labels:
  - console
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore latest-dev Console state tests and make runtime hook binding safe when a screen shell has not yet constructed every controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bare Console state serialize/restore tests do not bind mounted view hooks before their controllers exist
- [x] #2 Mounted Console hook binding preserves the retrieval capture callback
- [x] #3 Focused Console native state tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the state-restore failures from the bare Console screen helper.
2. Give the intentionally partial screen shell an explicit runtime before it assigns property-backed handles, avoiding mounted view-hook binding.
3. Run the isolated failures and focused state-restore subset.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture lifecycle repair that preserves existing runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- The bare `ChatScreen.__new__` fixture now installs an explicit `ConsoleRuntime` before assigning the property-backed chat store. This keeps deliberately partial test shells out of mounted view-hook binding while leaving production hook requirements unchanged.
- Verified the 12 previously failing Console state serialize/restore tests pass together. Mounted runtime hook tests also remain green.
- ADR required: no. This is a test-fixture lifecycle repair with no production boundary change.
<!-- SECTION:NOTES:END -->
