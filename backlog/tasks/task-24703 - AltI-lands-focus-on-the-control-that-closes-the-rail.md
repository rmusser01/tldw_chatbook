---
id: TASK-24703
title: Alt+I lands focus on the control that closes the rail
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 06:18'
updated_date: '2026-08-30 06:24'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-30
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-24604's action correctly moves focus into the rail on open, but CONSOLE_FOCUS_TARGETS_BY_PANE resolves console-right-rail to (console-inspector-rail-collapse, ...), so the caret arrives on the collapse button and the first Enter closes the pane the user just opened. Separately, at 80x24 the footer truncates and drops the 'Alt+I inspect' hint - which is the width where Alt+I is the ONLY route in, because the edge handle is hidden there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening the rail with its shortcut places focus on a non-destructive target
- [ ] #2 The Inspect shortcut hint survives footer truncation at the widths where it is the only route into the rail
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two halves.

Focus target: TASK-24604's action moved focus into the rail (right instinct) but CONSOLE_FOCUS_TARGETS_BY_PANE resolves console-right-rail to the collapse button first, so the caret landed on the control that closes the pane the user just opened. The shortcut path now targets '#console-send-authority-summary' -- focusable, non-destructive, and the thing they came to read -- falling back to the pane's normal targets when it is not mounted.

Footer: AppFooterStatus degrades by keeping a PREFIX of the hint list, so 'Alt+I inspect' at position 9 of 11 was the first thing dropped as width fell -- at exactly the width where the rail's edge handle is also hidden and Alt+I is the ONLY route in. Below the single-pane threshold the hint is now PROMOTED to the front rather than merely present, the same treatment the focus toggle already gets for the same reason. CONSOLE_WORKBENCH_SHORTCUTS is unchanged at normal widths, so the existing footer contract test still holds.

Verified live at 80x24: footer reads 'Ctrl+Shift+F focus | Alt+I inspect | F6 next pane | F1 · Ctrl+P · Ctrl+Q'.

Caught while editing: my first pass removed Alt+I from the base tuple instead of reordering a copy, which would have dropped the hint at every normal width.

Modified: UI/Screens/chat_screen.py, Tests/UI/test_console_inspector_keyboard_route.py.
<!-- SECTION:NOTES:END -->
