---
id: TASK-26025
title: 'Scheduling: scheduler liveness heartbeat'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - scheduling
  - ops
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A dead scheduler loop is indistinguishable from an idle one. Verified on origin/dev: a named grep for heartbeat across tldw_chatbook/Scheduling returns zero; the loop reports unhandled handler types at startup and increments scheduler_tasks_unhandled (Scheduling/scheduler/loop.py:349-362), but once running there is no liveness signal - if the loop dies, reminders simply stop and nothing says so. Hermes persists tick heartbeat, last-success age and last error, so its status command can state that the scheduler has not ticked in three hours.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each scheduler tick records a durable timestamp
- [ ] #2 The Schedules surface shows scheduler liveness, and a stale heartbeat is visibly distinct from an empty queue
- [ ] #3 The last error encountered by the loop is retained and surfaced rather than only logged
- [ ] #4 Staleness is judged against the configured poll interval, so a long interval does not read as a stall
- [ ] #5 The heartbeat write is cheap enough to run every tick without measurable overhead - measured and recorded
- [ ] #6 A never-started scheduler is distinguishable from a stalled one
<!-- AC:END -->
