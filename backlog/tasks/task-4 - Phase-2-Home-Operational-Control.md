---
id: TASK-4
title: 'Phase 2: Home Operational Control'
status: In Progress
assignee: []
created_date: '2026-05-03 14:47'
updated_date: '2026-05-05 00:19'
labels:
  - unified-shell
  - phase-2
  - home
  - control-surface
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Home a real dashboard and control surface for status, notifications, schedules, and active agent work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Home controls route to real services or explicit adapters.
- [ ] #2 Approve, reject, pause, resume, retry, and open-detail workflows are verified in the running app.
- [ ] #3 Unavailable states remain honest and recoverable.
- [ ] #4 Durable QA summary exists under Docs/superpowers/qa/unified-shell/phase-2/.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Treat TASK-4.1 through TASK-4.7 as implemented Home operational-control slices.
2. Keep parent TASK-4 open until TASK-4.8 completes maturity-gate QA in the running app.
3. Use TASK-4.8 to decide whether Home approve reject pause resume retry and open-detail workflows are verified or need explicit follow-up blockers.
<!-- SECTION:PLAN:END -->
