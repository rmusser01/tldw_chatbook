---
id: TASK-38532
title: Expose workflow setup and phase controls in Console
status: To Do
created_date: 2026-11-01 17:31
dependencies:
- TASK-38531
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-9
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 9 of the accepted native shared-worktree workflow design so expose workflow setup and phase controls in Console. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native workflow commands/setup/status/history expose exact revisions, current scope, limits, dirty input and change-review readiness without clearing drafts.
- [ ] #2 Approval, revision, Pause, Stop, final acceptance and explicit Resume/recovery use their exact durable owners and remain usable across navigation.
- [ ] #3 Existing fleet/Changes UI, finite DB offload/subscription patterns, design tokens, keyboard/focus and narrow/wide states are reused and verified.
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
