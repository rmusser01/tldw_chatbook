---
id: TASK-38531
title: Schedule approved phases through the existing Console runtime
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-11-01 23:15
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect native bridge/controller, activity, instruction discovery, physical settlement and startup recovery seams; resolve source-backed integration gaps before production edits.
2. Add real native integration rig and failing dispatch/scheduling/admission/settlement journeys.
3. Deliver and independently gate internal slice 8A: recovery denial/release events and separate physical guidance footprints; retain whole-task RED journeys and all unrelated work.
4. Deliver and gate focused 8B: extract genuine shared native bridge preparation/service lifetime, one registry/auth/seed composition and ordinary parity; do not duplicate the constructor or invent a parent turn.
5. Deliver subsequent narrowly scoped snapshot/outer-admission and phase-authority/continuation foundations, then event-driven coordinator integration. Preserve actual Change Review consent/phase windows and physical publication lifetimes; current Resume targets/tools and exact Pause/Stop semantics remain required.
6. Run the real dispatch/scheduling/admission/settlement journeys and named affected native tests; final all-commit Task8 review uses original BASE 1b3f64aa59ae2196d92bf0d486df55e2b30c1677. No partial gate completes parent AC.
7. Scoped static/preservation/self-review and final report. Leave parent status/AC pending full independent integration gate.
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Direct implementation with source-backed private producer/physical/continuation rulings recorded in task-8-context.md; decomposition reduces uncertain combined refactoring without reducing accepted scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
