---
id: TASK-38528
title: Enforce workflow task scope at native tool invocation
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 5 of the accepted native shared-worktree workflow design so enforce workflow task scope at native tool invocation. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every workflow task uses actual run-bound scope and registry ownership through discovery, runtime callbacks, review and invocation.
- [ ] #2 Alternate writers, root aliases, shell/session grants, progressive loading, skills and recursive task/new-chat routes cannot bypass captured task mode.
- [ ] #3 Opaque mode retains existing permissions/arming/trust and never claims OS confinement; denied/outside-scope work creates no effect.
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
