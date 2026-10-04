---
id: TASK-34382
title: >-
  File Notes replica retention policy: checkpoint cap, tombstone and revisions expiry
status: To Do
assignee: []
created_date: '2026-10-04'
labels:
  - notes
  - library
  - file-notes
dependencies: []
priority: high
---

## Description

Reslice of the superseded TASK-399 B-phase under the SHIPPED ADR-029 design
(one SQLite replica, disk authority; see the 2026-10-04 TASK-399 arc
reconnaissance). Without retention the replica grows unboundedly: checkpoints accumulate per session forever and tombstones/revisions never expire. This task adds bounded retention inside `file_notes_replica.py`.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Per-note checkpoint cap (order ~50) evicting oldest beyond the cap
- [ ] Tombstone and revision expiry (~30 days) with cleanup invoked on root change and/or session end
- [ ] Protected paths and the most recent tombstone are never evicted by the policy
- [ ] Retention is test-pinned (counts before/after, protected-preservation, recency)
<!-- AC:END -->

## Implementation Plan (to be added when claimed)

