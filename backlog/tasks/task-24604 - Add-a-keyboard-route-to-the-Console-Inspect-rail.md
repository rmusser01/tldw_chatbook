---
id: TASK-24604
title: Add a keyboard route to the Console Inspect rail
status: To Do
assignee: []
created_date: '2026-08-30 00:53'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Inspect rail defaults closed while the left rail defaults open. action_focus_next_workbench_pane excludes non-displayed panes and the collapsed handle is not in CONSOLE_FOCUS_REGISTRY.pane_order, so F6 cannot reach the rail in its shipping state. No Binding references it, CONSOLE_WORKBENCH_SHORTCUTS has no entry for it, and the command palette has none. Console already binds Alt+M for the model popover and Alt+W for the workspace switcher; the primary inspection surface has nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single keystroke opens the Inspect rail and places focus inside it from anywhere in Console
- [ ] #2 The same keystroke collapses the rail and returns focus to a sensible pane
- [ ] #3 The shortcut appears in the Console workbench shortcut list so the footer and F1 advertise it
- [ ] #4 A collapsed rail still occupies an F6 pane stop
<!-- AC:END -->
