---
id: TASK-38529
title: Share the existing fleet child launch path without behavior drift
status: Done
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
updated_date: 2026-11-01 21:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 6 of the accepted native shared-worktree workflow design so share the existing fleet child launch path without behavior drift. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary spawn and continuation retain routing, frozen targets, hooks, instructions, permission stamps, accounting and physical/client cleanup.
- [x] #2 A host-only child-launch seam accepts captured task launch custody without a bare thread or parallel execution owner.
- [x] #3 Inline/threaded behavior and failure/refund/settlement paths remain independently verified against existing fleet regressions.
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
Shared the existing native child composition and fleet launch/settlement tail. Original private workflow custody, activity-before-capacity, original reservation/run identity and native commit stay authoritative. Frozen task-only messages, explicit instruction snapshots, one native service per workflow task, duplicate fences and owned-log finalization preserve the original row/client/owner. Genuine fleet capacity waits retain the same preparation/allowance and optional writer; typed waits permit coordinator pending state without parsing error strings. Unknown/accepted launch states remain fenced.

Independent spec+quality review found and verified the fleet-capacity retry correction; scoped re-review passed. Full named qualification executed422 passing bodies with4 inherited isolation failures/4teardown, reproduced through read-only exact-BASE consumer import. Final owned-log covering48 passed; fix21host+7ordinary=28passed plus4 original allowance controls. Original profile failures and narrow supported-profile qualification remain separate. Scoped static checks and recorded unrelated-file/index preservation passed. Existing warnings remain disclosed; no full suite/live provider/GUI claim.

ADR required: no (existing ADR applies)
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Implements accepted native child-launch boundary with source-backed private preparation/wait contracts. Updated architecture and observed testing lesson. Runtime/coordinator bridge integration and required-result settlement remain later slices; workflow hook/script guards remain. Full evidence/contracts in .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-6-report.md. Commits e8a77b4db6 ande9cc2d2f39.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shared native composition/launch custody passed independent spec+quality review after the fleet-capacity retry fix. Targeted evidence and inherited limits are retained; required-result/coordinator integration follows in the existing plan.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
