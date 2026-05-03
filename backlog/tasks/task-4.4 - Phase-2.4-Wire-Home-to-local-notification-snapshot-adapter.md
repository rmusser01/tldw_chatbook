---
id: TASK-4.4
title: 'Phase 2.4: Wire Home to local notification snapshot adapter'
status: Done
assignee: []
created_date: '2026-05-03 18:19'
updated_date: '2026-05-03 18:22'
labels:
  - unified-shell
  - phase-2
  - home
  - notifications
  - adapter
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect Home to a synchronous local notification snapshot so the dashboard reflects real unread notification state without pretending generic notifications are controllable active runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home adapter can derive unread notification count from the local client notification service.
- [x] #2 Home Attention section exposes unread notifications without creating approval or active-run controls.
- [x] #3 Next-best action still prioritizes model readiness, approvals, failed schedules, and active work before notification review.
- [x] #4 Home falls back safely when no notification service is available or snapshot collection fails.
- [x] #5 Focused tests and Phase 2 QA evidence verify the adapter state, screen rendering, and backlog tracking.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing adapter and dashboard tests for local unread notification snapshots.
2. Implement the smallest Home dashboard state and adapter changes needed to expose unread notifications without live-work controls.
3. Wire TldwCli to use the local notification snapshot adapter after notification services initialize.
4. Add Phase 2 QA evidence and tracking updates.
5. Run focused Home adapter, dashboard, screen, and tracking verification plus git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a local notification snapshot adapter for Home that reads unread local notifications from `ClientNotificationsService.list_queue`, exposes `notification_count` in Home dashboard state, and wires `TldwCli` to use the adapter after notification services initialize. Dashboard copy now surfaces unread notifications and can recommend notification review after stronger readiness and live-work blockers, while generic notifications still do not create approval, pause, retry, detail, or Console controls. Added focused adapter, state, mounted Home screen, app wiring, and Phase 2 tracking tests plus QA evidence for `TASK-4.4`.
<!-- SECTION:NOTES:END -->
