---
id: TASK-38527
title: Preserve observed-content preconditions through edits and patches
status: To Do
created_date: 2026-11-01 17:30
dependencies:
- TASK-38524
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-4
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 4 of the accepted native shared-worktree workflow design so preserve observed-content preconditions through edits and patches. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 File edits and patches carry the digest of the actual read bytes through the existing atomic writer, refusing observed concurrent replacement.
- [ ] #2 Workflow writes require present-content or absent-file preconditions and cannot degrade to blind overwrite on missing/evicted stamps.
- [ ] #3 Existing local/remote worker, exclusion, permission and ordinary caller behavior retains its documented contract; partial multi-file effects remain honest.
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
