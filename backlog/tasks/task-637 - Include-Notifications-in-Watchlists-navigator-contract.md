---
id: TASK-637
title: Include Notifications in Watchlists navigator contract
status: Done
assignee:
  - '@codex'
created_date: '2026-07-25 19:37'
updated_date: '2026-07-25 19:38'
labels:
  - watchlists
  - ui
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the Watchlists navigator unit contract with the Notifications section restored by the current destination implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The navigator test expects all six current sections in production order.
- [x] #2 Notifications remains explicitly covered rather than weakening the test to a derived count.
- [x] #3 The focused navigator test and resumed post-UI block pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the five-versus-six failure and verify Notifications is an intentional production section with destination coverage.
2. Add the Notifications id to the explicit ordered navigator assertion.
3. Run the focused Watchlists tests, resumed post-UI block, and static checks.

ADR required: no
ADR path: N/A
Reason: This is a stale unit-test oracle following the existing Notifications destination behavior; no navigation architecture changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Extended the explicit ordered id assertion from five rows to all six current
  rows, including `nav-notifications`, and removed the test's unused `Button`
  import found by the static gate.
- The exact navigator failure passed, followed by 112 focused
  server/GitHub/Watchlists tests and the complete 459-test resumed post-UI
  block.
- Ruff, formatting, and diff checks passed. No production behavior changed.
<!-- SECTION:NOTES:END -->
