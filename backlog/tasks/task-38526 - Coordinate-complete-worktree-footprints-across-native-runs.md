---
id: TASK-38526
title: Coordinate complete worktree footprints across native runs
status: To Do
created_date: 2026-11-01 17:30
dependencies:
- TASK-38524
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-3
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 3 of the accepted native shared-worktree workflow design so coordinate complete worktree footprints across native runs. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical aliases, nested roots and read/write dependencies share correct nonblocking activity claims.
- [ ] #2 Opaque-capable work claims runtime-wide exclusivity before root-specific reservations; refusal creates no partial-lock deadlock.
- [ ] #3 Claims remain held until positive physical drain and conclusive durable settlement, including ordinary runs and survivors.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
