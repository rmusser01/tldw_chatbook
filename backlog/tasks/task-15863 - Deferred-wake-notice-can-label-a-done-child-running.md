---
id: TASK-15863
title: Deferred wake notice can label a done child 'running'
status: To Do
assignee: []
created_date: '2026-08-13 21:44'
labels:
  - fleet
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-2 Task 7 live: a wake deferred behind a non-empty composer draft (scenario 4) composed its notice with '[a687d2cb…] researcher — running — task: …' for a child whose agent_runs row had been 'done' for a full minute, while delivering that child's complete result in the same notice. The immediate-path wakes (#1-#3) and the post-restart poked delivery all said 'done'. compose_wake_notice takes status from the row dict _rows_for returns, which reads runs_db.get_run fresh — so either the read rode a connection holding a stale WAL snapshot (the held-connection trap from the defer-past-first-paint series), or a stand-in/registry status leaked through on this path. Diagnosis is part of the task; the notice is the honesty surface the supervisor acts on, so a wrong status word is a real defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A wake notice composed at delivery time reports the child's terminal status as of that read, on the immediate, deferred, and mount-claim paths
- [ ] #2 The mechanism that produced the stale word is identified and pinned by a test that fails on it
<!-- AC:END -->
