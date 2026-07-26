---
id: TASK-713
title: Workspace creation and activation changes are silent across Console and Library
status: To Do
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
- [ ] #1 Creating a workspace surfaces a visible confirmation naming the created workspace and stating it is now active
- [ ] #2 Any active-workspace change not initiated in the switcher modal surfaces a visible cue at the moment it happens
- [ ] #3 Cross-screen activation from Library is announced on the Library screen where the click happened
<!-- AC:END -->
