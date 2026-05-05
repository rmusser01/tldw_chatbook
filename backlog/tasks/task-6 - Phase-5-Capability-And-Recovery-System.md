---
id: TASK-6
title: 'Phase 5: Capability And Recovery System'
status: Done
assignee: []
created_date: '2026-05-03 14:47'
updated_date: '2026-05-05 05:25'
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
- [x] #1 Shared capability-state taxonomy exists and is applied to remaining blocker states.
- [x] #2 Blocked states tell users what is unavailable, why, and what to do next.
- [x] #3 Running-app QA verifies understandable recovery paths.
- [x] #4 Durable QA summaries exist under Docs/superpowers/qa/unified-shell/phase-5/.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create child tasks for Phase 5 in dependency order, starting with a shared recovery taxonomy.
2. Apply the taxonomy to blocker-state families in narrow implementation slices.
3. Replay running-app QA for blocked states before marking Phase 5 verified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Phase 5 recovery closeout through TASK-6.5. TASK-6.1 defines the shared recovery taxonomy, TASK-6.2 applies it to shell destination blockers, TASK-6.3 applies it to runtime-policy denials, TASK-6.4 applies it to optional-dependency blockers, and TASK-6.5 replays maturity-gate QA. The verified recovery fields cover what is unavailable why it is unavailable next action recovery target authority owner stable selector and disabled tooltip copy. Phase 5 is verified; live server/auth execution and full optional-extra runtime coverage remain Phase 6 or future service-depth residual risk.
<!-- SECTION:NOTES:END -->
