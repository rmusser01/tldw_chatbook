---
id: TASK-24604
title: Add a keyboard route to the Console Inspect rail
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:53'
updated_date: '2026-08-30 01:50'
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
- [x] #1 A single keystroke opens the Inspect rail and places focus inside it from anywhere in Console
- [x] #2 The same keystroke collapses the rail and returns focus to a sensible pane
- [x] #3 The shortcut appears in the Console workbench shortcut list so the footer and F1 advertise it
- [x] #4 A collapsed rail still occupies an F6 pane stop
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Binding('alt+i', 'toggle_console_inspector_rail', 'Inspect') alongside the existing alt+m / alt+w, an entry in CONSOLE_WORKBENCH_SHORTCUTS so the footer and F1 teach it, and a collapsed stand-in so F6 keeps its Inspector stop.

Design deviation from the task text, deliberate. The AC proposed adding 'console-inspector-rail-handle' to CONSOLE_FOCUS_REGISTRY.pane_order. That is the wrong mechanism: _console_workbench_focus_id_for_widget checks pane_order BEFORE CONSOLE_FOCUS_PANE_FOR_WIDGET, so promoting the handle to a pane would make focus inside the collapsed rail report the handle rather than its logical pane and silently change TASK-2154.11's documented between-panes behaviour. Instead CONSOLE_PANE_COLLAPSED_STAND_IN maps each rail pane to its collapsed widget, and two places consult it: _console_pane_is_reachable (so F6's hidden-set asks 'can I land here', not 'is the rail open') and _console_workbench_focus_targets (so the handle is the pane's last-resort focus target). The test asserting the pane_order approach was rewritten to assert the mechanism actually used and to pin the TASK-2154.11 invariant it protects.

The action opens AND focuses: an accelerator that reveals a pane but leaves the caret in the composer sends the user back to the mouse, which is the problem. Closing returns focus to the composer rather than stranding it on a widget about to be hidden.

Pre-existing dev failures verified identical on a pristine origin/dev worktree, not caused here: 2 in test_console_inspector_compact_access.py.

Modified: tldw_chatbook/UI/Screens/chat_screen.py, Docs/User_Guide/console/sessions-tabs-workspaces.md, Tests/UI/test_console_inspector_keyboard_route.py (new).
<!-- SECTION:NOTES:END -->
