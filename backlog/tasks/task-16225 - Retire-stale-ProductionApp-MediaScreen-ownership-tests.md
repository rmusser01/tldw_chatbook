---
id: TASK-16225
title: Retire stale ProductionApp MediaScreen ownership tests
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:47'
updated_date: '2026-08-14 08:57'
labels:
  - testing
  - navigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep ProductionApp integration evidence aligned with TASK-2851 now that the media route is owned by Library rather than the retired standalone MediaScreen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ProductionApp route inventory asserts media resolves to the canonical Library screen and tab
- [x] #2 Obsolete production-route tests for the unreachable standalone MediaScreen are removed without weakening direct legacy unit coverage
- [x] #3 Focused ProductionApp ownership and navigation tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the failures are caused by the TASK-2851 media-to-Library route alias.
2. Remove the unreachable MediaScreen ProductionApp route suite and update the route ownership manifest to Library.
3. Run focused navigation/ownership tests, static checks, and review the diff.

ADR required: no
ADR path: N/A
Reason: This is test reconciliation with the existing TASK-2851 route-retirement decision; no boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the obsolete 1,507-line ProductionApp suite whose premise contradicted TASK-2851 by requiring NavigateToScreen("media") to mount the retired MediaScreen. Updated the production route maturity manifest to assert the canonical LibraryScreen/TAB_LIBRARY owner and Media rail row on both visits. The full route/privacy tour and focused media navigation regressions pass; Ruff lint/format and diff checks pass. Direct MediaScreen save/restore remains covered by its dedicated unit suites. ADR required: no (implements the existing TASK-2851 route-retirement decision).
<!-- SECTION:NOTES:END -->
