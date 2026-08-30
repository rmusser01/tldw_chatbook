---
id: TASK-24610
title: Sources names two different things in the same Inspect rail
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 01:45'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The staged-context tray heading Sources means staged context references. The Sources row under Source Readiness means retrieval status and reads 'Sources: not staged'. The pinned authority row means the first. The Run recipe line's sources summary means the second. All four are visible at once in a 33-column column, which makes this the largest comprehension cost in the rail. It is a rename, not a rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One noun in the Inspect rail refers to exactly one concept
- [ ] #2 The retrieval status row is named for retrieval rather than for sources
- [ ] #3 Sources means staged context consistently across the tray, authority row, status chip and rail handle badge
<!-- AC:END -->
