---
id: TASK-26046
title: Defer permission-summary imports from UI-ready startup
status: Done
assignee: []
created_date: '2026-09-01 02:24'
updated_date: '2026-09-01 02:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the existing UI-ready module ceiling after permission-request summaries made their LLM and trace dependency graph resident before the first interactive frame.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current dev's UI-ready module census passes without raising its ceiling.
- [x] #2 Permission summaries retain their existing behavior.
- [x] #3 Focused permission-summary and startup-budget tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the eager import path against the failing dev census\n2. Defer permission-summary service imports to the actions that use them\n3. Add focused regression coverage for the lazy boundary\n4. Run focused behavior, import-budget, lint, and artifact checks\n5. ADR required: no; ADR path: N/A; reason: direct regression fix implementing existing ADR-090 and ADR-097 boundaries
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred permission-summary, terminal support, and trace disclosure imports from UI-ready startup; added focused lazy-boundary coverage; verified the 972-module ceiling plus focused behavior, inspector, terminal, performance, lint, compile, diff, and diagnostic inventory checks. ADR required: no; implements ADR-090 and ADR-097.
<!-- SECTION:NOTES:END -->
