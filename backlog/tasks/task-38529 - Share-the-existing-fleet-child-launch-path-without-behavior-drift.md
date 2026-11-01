---
id: TASK-38529
title: Share the existing fleet child launch path without behavior drift
status: In Progress
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
updated_date: 2026-11-01 21:20
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect native spawn, continuation, client and owner settlement; retain the required original targeted baseline.
2. Resolve the frozen native preparation-envelope contract before production edits. Extract the existing child composition/launch path with explicit captured inputs and preserve ordinary inline/fleet semantics.
3. Wire original private workflow custody, worktree activity, the same reserved child allowance and run identity through native commit/binding; refuse duplicate dispatch before any row/client/capacity mutation.
4. Add focused positive, negative, concurrent-duplicate and lifetime tests; run named fleet/routing/settlement regressions and scoped static checks only.
5. Self-review, document exact evidence and limitations, and commit only owned paths. Leave acceptance criteria/status pending independent review.
ADR required: no (existing ADR applies)
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: This slice implements the approved native shared-child-launch boundary; any new preparation-boundary decision is raised before production edits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Shared the existing native child composition and fleet launch/settlement tail with private workflow preparation. Original Task5 custody, activity-before-capacity, reserved allowance/run identity and native commit remain authoritative. Frozen task-only messages and explicit instruction snapshots survive caller mutation; one native service per task and duplicate fences protect the original row, client and owner, including the direct no-log route. Task-owned logs reuse native finalization once; shared parent logs remain parent-owned.

Extended real native host/duplicate/failure/client-drain tests, ordinary owner/continuation assertions and exact hook lifecycle counts. The full named regression run executed422 passing bodies with four inherited isolated-worktree scope failures; exact BASE consumer import reproduced those four without replacing shared source. Original source-profile failures and exact-node supported-profile qualifications remain separately recorded. Scoped lint/format/compile and preservation checks passed; legacy lint/format findings were not broadly rewritten. The task remains In Progress with AC pending independent review.

ADR: backlog/decisions/285-native-phased-agent-workflows.md. Architecture and the observed source-profile masking lesson updated. Full commands/results/contracts/limits are in .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-6-report.md. Real coordinator wiring and required-result settlement remain with their existing subsequent implementation slices; workflow hooks/scripts stay unsupported.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
