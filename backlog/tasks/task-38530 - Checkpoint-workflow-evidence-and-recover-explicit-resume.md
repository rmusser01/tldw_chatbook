---
id: TASK-38530
title: Checkpoint workflow evidence and recover explicit resume
status: To Do
created_date: 2026-11-01 17:31
dependencies:
- TASK-38524
- TASK-38525
- TASK-38526
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-7
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 7 of the accepted native shared-worktree workflow design so checkpoint workflow evidence and recover explicit resume. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Required handoffs/checkpoints persist independently of optional logs and remain protected from pruning while active or uncertain.
- [ ] #2 Exact manifests/verifier identities, typed process results and current binding/target evidence govern phase advancement and final review.
- [ ] #3 One startup recovery audit retains uncertain charges and requires explicit Resume; quality review never clears cleanup uncertainty or authorizes replay.
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
