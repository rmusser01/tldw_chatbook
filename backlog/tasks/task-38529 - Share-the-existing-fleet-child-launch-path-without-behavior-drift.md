---
id: TASK-38529
title: Share the existing fleet child launch path without behavior drift
status: To Do
created_date: 2026-11-01 17:31
dependencies:
- TASK-38524
- TASK-38525
- TASK-38528
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-6
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
updated_date: 2026-11-01 17:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 6 of the accepted native shared-worktree workflow design so share the existing fleet child launch path without behavior drift. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ordinary spawn and continuation retain routing, frozen targets, hooks, instructions, permission stamps, accounting and physical/client cleanup.
- [ ] #2 A host-only child-launch seam accepts captured task launch custody without a bare thread or parallel execution owner.
- [ ] #3 Inline/threaded behavior and failure/refund/settlement paths remain independently verified against existing fleet regressions.
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
