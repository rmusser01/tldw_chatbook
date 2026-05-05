---
id: TASK-6
title: 'Phase 5: Capability And Recovery System'
status: In Progress
assignee: []
created_date: '2026-05-03 14:47'
updated_date: '2026-05-05 02:32'
labels:
  - unified-shell
  - phase-5
  - capability-state
  - recovery
dependencies: []
documentation:
  - Docs/Design/master-shell-design-system-contract.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Systematize missing dependency, auth, server, runtime, and policy states so blocked workflows are understandable and recoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared capability-state taxonomy exists and is applied to remaining blocker states.
- [ ] #2 Blocked states tell users what is unavailable, why, and what to do next.
- [ ] #3 Running-app QA verifies understandable recovery paths.
- [ ] #4 Durable QA summaries exist under Docs/superpowers/qa/unified-shell/phase-5/.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create child tasks for Phase 5 in dependency order, starting with a shared recovery taxonomy.
2. Apply the taxonomy to blocker-state families in narrow implementation slices.
3. Replay running-app QA for blocked states before marking Phase 5 verified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Phase 5 with TASK-6.1, which defines the shared recovery taxonomy and tracking contract. TASK-6.2 applies the taxonomy to the first shell destination blocker family: ACP runtime configuration, Schedules empty active-run, Workflows empty active-run, and Artifacts empty Chatbook states. Parent Phase 5 remains In Progress until later slices apply the taxonomy to remaining runtime-policy and optional-dependency blocker families and running-app QA verifies recovery paths.
<!-- SECTION:NOTES:END -->
