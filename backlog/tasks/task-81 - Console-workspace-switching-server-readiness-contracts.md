---
id: TASK-81
title: Console workspace switching server-readiness contracts
status: To Do
labels:
- console
- workspaces
- server-readiness
- ux
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare Console workspace switching for future tldw_server and ACP handoff support without claiming or implementing a sync engine. Console should keep local workspace switching usable now, preserve explicit staging rules, avoid hiding global Library or Notes content, and expose honest unavailable/readiness states for future server workspace and ACP task/run package migration paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console local workspace switching remains backed by the local registry fallback and does not depend on tldw_server sync.
- [ ] #2 Workspace switching changes Console operating context and conversation scope, but Library/Notes/global browse visibility remains intact.
- [ ] #3 Console clearly distinguishes local-only, server-unavailable, remote-only, conflict, and runtime-missing workspace states using existing workspace authority/sync language.
- [ ] #4 Console supports explicit copy/reference/metadata-only/local-only handoff eligibility states for sources and conversations before staging them into the active workspace.
- [ ] #5 ACP task/run package handoff is represented as a future migration target with visible unavailable, readiness, failure, and audit details.
- [ ] #6 No background sync engine is implemented or implied; server-backed hydration remains behind an adapter boundary that can be wired only when the server API is available.
- [ ] #7 Focused regressions cover local fallback, server-unavailable states, cross-workspace gating, and ACP handoff blocked/ready states. Actual CDP screenshots are captured before approval.
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
