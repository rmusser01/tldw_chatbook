---
id: TASK-38524
title: Persist bounded workflow plans and private task results
status: To Do
created_date: 2026-11-01 17:30
dependencies:
- TASK-38523
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-1
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 1 of the accepted native shared-worktree workflow design so persist bounded workflow plans and private task results. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Strict plans enforce ordered dependencies, captured scope/targets, byte and count limits without execution authority.
- [ ] #2 Workflow creation, revision control and private result reservations are durable, scoped, idempotent and preserve accounting tombstones.
- [ ] #3 Real migrations/reopen and exact recovery/backup schema validation preserve existing run/Goal/fleet data and keep restored workflows inert.
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
