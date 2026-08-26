---
id: TASK-16226
title: Bound detached Personas lore mount
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:50'
updated_date: '2026-08-14 08:55'
labels:
  - personas
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent deferred Personas lore initialization from crashing when navigation detaches the screen during widget mount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A detached lore detail widget exits mount initialization without querying absent descendants
- [x] #2 A normally mounted lore detail widget still initializes both tables
- [x] #3 Focused Personas deferred-view and ProductionApp route tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression for detached mount initialization.
2. Add the smallest attachment guard before descendant queries.
3. Run focused widget, deferred-view, and ProductionApp route tests plus static checks.

ADR required: no
ADR path: N/A
Reason: This is a lifecycle bug fix within the existing deferred Personas view boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a detached-mount guard to PersonasLoreDetailWidget before descendant queries and a focused regression proving the callback is a no-op after teardown. Normal mounted Lore initialization and all deferred-center-view tests remain green; the ProductionApp route tour now completes without the prior mount crash. Ruff lint and py_compile pass. Ruff format remains the identical pre-existing baseline drift in both touched legacy files; no unrelated whole-file formatting churn was introduced. ADR required: no (existing deferred-view boundary unchanged).
<!-- SECTION:NOTES:END -->
