---
id: TASK-38533
title: Qualify shared-worktree workflows across execution and restart
status: To Do
created_date: 2026-11-01 17:31
dependencies:
- TASK-38531
- TASK-38532
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-10
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 10 of the accepted native shared-worktree workflow design so qualify shared-worktree workflows across execution and restart. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A1–A12 have retained evidence from real files/Git/SQLite, native admission, held-worker races and fresh-process recovery controls.
- [ ] #2 A real provider performs an authorized correction between failed and passing targeted checks with exact manifests, process results and phase approvals.
- [ ] #3 Living architecture/user/Settings/recovery docs describe only qualified behavior; unresolved bootstrap, platform or uncertainty limitations remain explicit.
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
