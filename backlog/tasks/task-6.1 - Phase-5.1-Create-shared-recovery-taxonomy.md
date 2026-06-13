---
id: TASK-6.1
title: Phase 5.1 Create shared recovery taxonomy
status: Done
assignee: []
created_date: '2026-05-05 02:31'
updated_date: '2026-05-05 02:32'
labels:
  - unified-shell
  - phase-5
  - capability-state
  - recovery
dependencies: []
parent_task_id: TASK-6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the shared shell capability-state and recovery taxonomy that later Phase 5 slices can apply to blocked dependency auth server runtime and policy states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recovery taxonomy documents canonical blocked-state categories and required user-facing fields
- [x] #2 Taxonomy maps to existing runtime-policy reason codes and destination recovery states
- [x] #3 Phase 5 roadmap README and parent task record the Phase 5.1 slice
- [x] #4 Regression test verifies the taxonomy artifact task and roadmap stay linked
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 5 tracking regression for the taxonomy artifact and TASK-6.1 linkage.
2. Create the shared recovery taxonomy document using existing runtime-policy reason codes and shell recovery states.
3. Update Phase 5 README and maturity roadmap to mark Phase 5 in progress and link TASK-6.1.
4. Run the focused Phase 5 tracking regression and documentation checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Defined the shared Phase 5 recovery taxonomy in Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md. The taxonomy maps shell blocked states to existing runtime-policy reason codes, domain edge contracts, and prior destination recovery evidence, then defines required user-facing fields for later application slices. Added a focused regression to keep the taxonomy artifact, Phase 5 README, roadmap, and task status linked.
<!-- SECTION:NOTES:END -->
