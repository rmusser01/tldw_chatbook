---
id: TASK-713
title: Workspace creation and activation changes are silent across Console and Library
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - library
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Creating a workspace from Console has no success notification (only error paths notify) and immediately activates the workspace with no name prompt. Clicking a conversation row in another workspace's group silently switches the active workspace with the status row typically scrolled out of view. Creating a workspace from Library silently retargets Console's active context cross-screen. Findings M1/M2; live evidence captures cap-04/05, cap-14/15, cap-23-25.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Creating a workspace surfaces a visible confirmation naming the created workspace and stating it is now active
- [x] #2 Any active-workspace change not initiated in the switcher modal surfaces a visible cue at the moment it happens
- [x] #3 Cross-screen activation from Library is announced on the Library screen where the click happened
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red tests: Console create notify, browser-row switch notify (+ no re-announce for active workspace), Library toast naming the Console retarget.
2. Add notifications at the three trigger points; green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Console create (`on_console_new_workspace`) now notifies "Created <name> and switched Console to it." after the sync sequence. Cross-workspace row activation (`_activate_console_workspace_for_browser_row`) notifies "Switched Console to <name>." only when the active workspace actually changed. Library create toast now reads "Created local workspace <name> and made it active; Console now targets it." (correction from the UAT report: this handler DID already toast - the live capture missed it - but the copy never stated the activation side effect). Tests: 2 new in test_console_new_workspace.py, 1 new in test_post_release_workspaces_library_depth.py.
<!-- SECTION:NOTES:END -->
