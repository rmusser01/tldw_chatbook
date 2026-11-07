---
id: TASK-38525
title: Admit workflow attempts with cumulative active-time budgets
status: Done
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
updated_date: 2026-11-01 19:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 2 of the accepted native shared-worktree workflow design so admit workflow attempts with cumulative active-time budgets. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed workflow phase/task claims enforce durable acceptance before billable work and at most one launch per accepted task attempt.
- [x] #2 Workflow active time charges parallel execution once and preserves consumed/uncertain charges across review, restart and live reductions.
- [x] #3 Workflow-specific settings use canonical validation/Settings and leave Goal/fleet elapsed deadlines and manual capacity policy unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: implements the approved workflow admission/accounting policy without changing its boundaries.
1. Audit passive workflow links and native automatic admission, registry, recovery and Settings consumers; preserve shared-worktree changes.
2. Add targeted failing SQLite phase/task admission, ownership, timing and policy tests; retain RED receipts.
3. Extend native attempt storage with the next additive migration, durable phase/task claims and private process-local authorization.
4. Add workflow-only cumulative active-interval accounting and workflow policy gates without changing Goal/fleet deadlines.
5. Add canonical workflow Settings using existing Save/Revert validation patterns; run targeted regression/static checks and self-review.
6. Record implementation evidence and commit only task-owned paths/hunks; leave task In Progress pending independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented workflow-specific native admission/accounting under ADR-285. Added typed planning/execution phase references, FULL acceptance, private live-owner/session custody, stable task child reservations/requested run IDs, and exact native run binding. Workflow union intervals retain clock ownership/charges through uncertainty and close only with positive drain plus durable results; planning publication settles atomically. Canonical Console behavior Settings uses independent finite workflow ceilings and Save/Revert. v26 constructor, standalone migration, installed recovery catalog/staging allowlist and semantic validators preserve v25 history.

Verification: 163 targeted tests passed (exit 0), including Goal/fleet budgets/deadlines, both mounted Settings flows, workflow history/migrations and complete archive/restore/rebackup. Original RED and intermediate receipts retained in .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-2-*.log. Ruff baseline comparison: zero introduced diagnostics across 22 owned source/test files (365 pre-existing); new files lint/format clean. Existing pytest garbage-directory cleanup warnings remain recorded.

Controller-approved boundaries: planning uses explicit purpose without a fake plan; WORKFLOW_PHASE controller submission fails closed until Task8. claim_workflow_task exposes reservation_id for existing AgentService automatic_child_reservation commit and launch_run_id for requested_run_id; duplicate claims never allocate another allowance. Physical drain is currently a trusted coordinator observation parameter, not yet wired to ExecutionOwner (Tasks7/8). Existing passive stop/save-plan evidence guards remain intact; pause/control CAS revokes dispatch. Actual catalog owners were recovery_operations/sqlite_validation; unrelated recovery_core modules were not edited. ADR required: no new ADR; implements backlog/decisions/285-native-phased-agent-workflows.md.

Task remains In Progress with AC pending the controller's independent review. Full implementation, ownership, source-backed deviations and Task5/6/7/8 interface handoff are in task-2-report.md.
Independent spec/quality gate passed after fix7d9a48f5e0 (base implementation387c79d000). Added durable within-phase quiescence and enforced explicit/provider/child/tool reopening without allowance refill; failed/cancelled planning now settles without obsolete plan publication, retaining charges and successful-publication CAS. Final fix124 covering checks passed, then2 affected test-style checks; zero introduced static findings. Initial163 targeted and final custody41 runs include v26 real archive/restore/rebackup and mounted Settings; original warnings retained. No full suite. Actual ExecutionOwner drain proof and positive scheduling remain Tasks7–8 under ADR-285.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workflow-specific durable admission, active-time allowance and canonical Settings implemented and independently reviewed. Controller dispatch remains fail-closed until runtime scheduling integration.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
