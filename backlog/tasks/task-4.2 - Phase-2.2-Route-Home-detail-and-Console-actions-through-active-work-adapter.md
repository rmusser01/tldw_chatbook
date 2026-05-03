---
id: TASK-4.2
title: 'Phase 2.2: Route Home detail and Console actions through active-work adapter'
status: Done
assignee: []
created_date: '2026-05-03 17:45'
updated_date: '2026-05-03 17:50'
labels:
  - unified-shell
  - phase-2
  - home
  - adapter
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Home detail and Console live-work actions behind the active-work adapter boundary so Home does not present direct navigation as service-backed control when no active-work payload exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home open-detail action delegates to the adapter for its target route or unavailable result.
- [x] #2 Home open-in-console action delegates to the adapter for actionable Console launch context or unavailable result.
- [x] #3 Adapter results preserve honest recovery copy when service-backed detail or Console payloads are unavailable.
- [x] #4 Focused unit and mounted Home tests cover detail and Console delegation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing adapter and mounted Home tests for open-detail and open-in-console delegation.
2. Extend the Home active-work adapter result contract only as far as needed for route and Console launch context.
3. Wire HomeScreen and app-level helpers so detail and Console actions use adapter outcomes instead of direct fallback navigation.
4. Add Phase 2 QA evidence and roadmap/task updates.
5. Run focused Home tests plus diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the Home active-work adapter contract with detail and Console actions, route and Console launch result fields, and honest unavailable defaults. Home buttons now call runtime hooks with target routes, app hooks delegate to the adapter, details navigate only on handled target results, and Console opens only with adapter-supplied launch payloads. Added focused unit, mounted Home, and Phase 2 evidence-link tests plus QA documentation.
<!-- SECTION:NOTES:END -->
