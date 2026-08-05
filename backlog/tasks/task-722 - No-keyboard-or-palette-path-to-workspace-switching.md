---
id: TASK-722
title: No keyboard or palette path to workspace switching
status: Done
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
- [x] #1 A keybinding opens the workspace switcher from the Console screen
- [x] #2 Command palette exposes switch-workspace (and create-workspace) entries
- [x] #3 The switcher modal is fully keyboard-operable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red tests: Alt+W opens switcher; palette lists switch/new workspace; modal keyboard-only round-trip.
2. Extract shared open/create methods, add binding + actions + palette entries + modal AUTO_FOCUS/arrow bindings; green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Alt+W binding (footer-visible, alongside Alt+M Model) -> action_open_console_workspace_switcher; extracted _open_console_workspace_switcher and _create_console_workspace so the rail buttons, binding, and palette share one path. ConsoleCommandProvider gains "Console: Switch workspace…" and "Console: New workspace". Switcher modal: AUTO_FOCUS on the first actionable option + up/down focus bindings; Enter selects, Escape cancels. Tests: Tests/UI/test_console_workspace_keyboard.py (3 tests) + existing palette/new-workspace suites green (66 total).
<!-- SECTION:NOTES:END -->
