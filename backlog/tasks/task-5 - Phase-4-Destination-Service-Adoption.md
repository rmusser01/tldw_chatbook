---
id: TASK-5
title: 'Phase 4: Destination Service Adoption'
status: Done
assignee: []
created_date: '2026-05-03 14:47'
updated_date: '2026-05-05 01:57'
labels:
  - unified-shell
  - phase-4
  - destinations
  - service-adoption
dependencies: []
documentation:
  - Docs/Design/master-shell-route-inventory.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn top-level wrappers into useful product surfaces by adopting real services and meaningful list, detail, and action flows where services exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each destination supports at least one meaningful end-to-end workflow or an honest recovery state.
- [x] #2 Skills, MCP, ACP, Library, Workflows, Schedules, Personas, Artifacts, W+C, and Settings ownership is explicit.
- [x] #3 Running-app QA verifies destination workflows rather than render-only behavior.
- [x] #4 Durable QA summaries exist under Docs/superpowers/qa/unified-shell/phase-4/.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Treat TASK-5.1 through TASK-5.5 as implemented destination service-adoption slices.
2. Keep parent TASK-5 open until TASK-5.6 completes maturity-gate QA in the running app.
3. Use TASK-5.6 to decide whether each top-level destination has a meaningful workflow or honest recovery state, and whether deeper service gaps need explicit follow-up blockers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed Phase 4 through TASK-5.6 maturity-gate replay. Destination wrappers now have durable QA evidence for meaningful local workflows, Console staging, explicit ownership, or honest recovery states across Skills, MCP, ACP, Library, Workflows, Schedules, Personas, Artifacts, W+C, and Settings. Full CRUD/server-parity depth remains tracked as later-phase residual risk, not a Phase 4 blocker.
<!-- SECTION:NOTES:END -->
