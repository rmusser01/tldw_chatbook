---
id: TASK-64
title: Implement Console workspace context rail
status: Done
labels:
- workspaces
- console
- pr-b
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR B from the workspace operating context plan: split the Console left rail into Staged Context and a read-only Convos & Workspaces panel backed by the workspace registry display state, without enabling workspace switching or hidden Library/Notes filtering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console exposes a second left-rail panel titled Convos & Workspaces while preserving the existing Staged Context tray.
- [x] #2 The workspace context rail renders honest missing-service, no-active-workspace, and active-workspace states from pure display-state logic.
- [x] #3 Change workspace remains disabled with recovery copy until active workspace switching is implemented in a later slice.
- [x] #4 Mounted UI regressions verify stable Console selectors and preserve the existing terminal-native workbench layout.
- [x] #5 Actual rendered screenshot/CDP evidence is captured before visual approval and PR closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review Console compose/widget patterns and existing workspace foundation APIs.
2. Write failing display-state tests for missing service, no active workspace, and active workspace states.
3. Write failing mounted Console tests for the two-section left rail selectors and disabled recovery states.
4. Implement pure workspace display state and the `ConsoleWorkspaceContextTray` widget.
5. Compose the left rail in `ChatScreen` without changing workspace switching, conversation persistence, or Library/Notes visibility behavior.
6. Run focused verification and capture actual rendered screenshot/CDP evidence before PR closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a pure Console workspace display-state builder that reads local registry state, runtime bindings, and conversation memberships without coupling Console rendering to storage details.
- Added a terminal-native `ConsoleWorkspaceContextTray` and split the Console left rail into staged context plus read-only workspace/conversation context.
- Kept `Change workspace` and `New conversation` visible but disabled with explicit recovery copy because action wiring belongs to later workspace slices.
- Added focused display-state and mounted UI regressions, updated Console geometry/parity tests for the split left rail, and captured an actual `textual-web`/CDP screenshot for approval.
- Addressed PR review feedback by wiring active conversation highlighting, adding tray state sync, rendering workspace text without Rich markup interpretation, logging degraded registry reads, and hardening empty membership/runtime states.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console now exposes a read-only workspace context rail alongside staged context, preserving the existing Console workbench while making the active workspace and associated conversations visible.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
