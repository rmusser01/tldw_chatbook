---
id: TASK-38528
title: Enforce workflow task scope at native tool invocation
status: Done
created_date: 2026-11-01 17:30
dependencies:
- TASK-38524
- TASK-38525
- TASK-38526
- TASK-38527
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-5
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
assignee:
- '@codex'
updated_date: 2026-11-01 20:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 5 of the accepted native shared-worktree workflow design so enforce workflow task scope at native tool invocation. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every workflow task uses actual run-bound scope and registry ownership through discovery, runtime callbacks, review and invocation.
- [x] #2 Alternate writers, root aliases, shell/session grants, progressive loading, skills and recursive task/new-chat routes cannot bypass captured task mode.
- [x] #3 Opaque mode retains existing permissions/arming/trust and never claims OS confinement; denied/outside-scope work creates no effect.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory actual effect routes and private phase/launch/activity contracts; confirm issuance/binding amendments before production edits.
2. Add negative dispatch tests with supported private-profile fixtures and meaningful RED evidence.
3. Bind immutable restrictions to native run/owner; compose discovery, review, callbacks and final invocation with existing permissions/hooks/writer.
4. Targeted regression/static checks, self-review and exact owned commit; preserve unrelated changes.
ADR required: no new ADR.
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: direct implementation of accepted native authority contracts. Leave In Progress and AC pending independent gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented original-phase private task preparation and once-only native run/capacity/worktree-claim binding. Discovery, callbacks, review and final provider invocation intersect captured tools/paths with current authority and the Task4 writer. Workflow effectful hooks and skill scripts explicitly refuse pending owned cleanup (ADR-285/spec amended). Final targeted verification: 208 passed/270 deselected; raw-shell/MCP36 passed; script/Goal33 passed. Zero introduced Ruff diagnostics. Full route/evidence inventory: .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-5-report.md. In Progress and AC pending independent gate.
Independent spec/quality gate passed after test-only fixdbf3f00e0f (basee4936326fb). Original private phase/task preparation binds once to actual native row, owner, runtime activity and captured provider/root; guards cover discovery, lazy instruction preflight, review, runtime callbacks and final post-wait invocation. Added7 exact custody negatives/matching controls;106 scope/activity checks pass. Initial final matrix51, combinednative208, raw/MCP36, scriptGoal33 pass; selections overlap. Zero introduced static findings. Hook/script unsupported routes are explicit in amended ADR285/spec, not false cleanup proof. Original source-lifetime failures/noisy receipts and temporary-base-swap audit limit remain disclosed, no evidence of lost work. Original staged patch+171 initial unowned hashes preserved. Task6 must retain original private preparation/reservation/run ID and actual claim through shared native launch; Task8 handles scheduling/settlement. No full-suite/live remote qualification claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native run-bound workflow task scope implemented and independently reviewed across supported invocation routes. Unowned effectful hooks and workflow scripts refuse explicitly; scheduling/positive shared launch remain later slices.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
