---
id: TASK-722
title: No keyboard or palette path to workspace switching
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - keyboard
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatScreen has no workspace-related binding and the command palette has no switch-workspace entry; Switch/New are mouse-only compact buttons. For a TUI the core context-switch operation should be reachable from the keyboard. Finding m6.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A keybinding opens the workspace switcher from the Console screen
- [ ] #2 Command palette exposes switch-workspace (and create-workspace) entries
- [ ] #3 The switcher modal is fully keyboard-operable
<!-- AC:END -->
