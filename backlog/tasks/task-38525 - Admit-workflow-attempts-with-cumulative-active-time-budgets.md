---
id: TASK-38525
title: Admit workflow attempts with cumulative active-time budgets
status: To Do
created_date: 2026-11-01 17:30
dependencies:
- TASK-38524
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-2
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 2 of the accepted native shared-worktree workflow design so admit workflow attempts with cumulative active-time budgets. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed workflow phase/task claims enforce durable acceptance before billable work and at most one launch per accepted task attempt.
- [ ] #2 Workflow active time charges parallel execution once and preserves consumed/uncertain charges across review, restart and live reductions.
- [ ] #3 Workflow-specific settings use canonical validation/Settings and leave Goal/fleet elapsed deadlines and manual capacity policy unchanged.
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
