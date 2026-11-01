---
id: TASK-38531
title: Schedule approved phases through the existing Console runtime
status: To Do
created_date: 2026-11-01 17:31
dependencies:
- TASK-38524
- TASK-38525
- TASK-38526
- TASK-38527
- TASK-38528
- TASK-38529
- TASK-38530
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-8
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 8 of the accepted native shared-worktree workflow design so schedule approved phases through the existing Console runtime. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read-only planning and approved task phases dispatch through typed native authority with per-session/global capacity and whole-chain accounting.
- [ ] #2 Host settlement events drive only approved eligible tasks; workflow-managed children do not independently trigger ordinary fleet wakes.
- [ ] #3 Ordinary/native opaque activity honors worktree claims; Pause/Stop/replan revoke launch authority and preserve user files and durable results.
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
