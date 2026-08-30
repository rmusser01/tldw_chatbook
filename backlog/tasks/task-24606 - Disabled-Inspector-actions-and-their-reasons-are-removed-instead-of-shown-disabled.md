---
id: TASK-24606
title: >-
  Disabled Inspector actions and their reasons are removed instead of shown
  disabled
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 02:46'
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
_button_for_action sets display none, width 0 and height 0 on a disabled action, and _widgets_for_action mounts its disabled_reason with display none and height 0 plus console-hidden-control. Authored strings such as 'No approval is pending.' and 'No Chatbook artifact is available.' are threaded through the ownership classifier and never rendered. The design system explicitly forbids hiding why an action is unavailable and names the inspector as one of the surfaces that must carry the reason. Actions also appear and disappear between turns, costing spatial memory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A disabled Inspector action stays mounted and visibly disabled rather than being removed from layout
- [x] #2 The reason an action is unavailable is readable without a mouse, for example carried in the label
- [x] #3 A disabled action label meets at least 3:1 against its own background, measured in a running terminal
- [x] #4 A test asserts the disabled action and its reason are present and displayed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Disabled Inspector actions are now shown disabled, with their reason readable, instead of being erased.

THREE layers were hiding the explanation, not the one the task named. Found by making the test pass rather than by reading:
1. _group_widgets dropped the whole GROUP when it had no rows and no ENABLED action -- so the outermost layer hid the reason even from a user who knew where to look. A group now also earns its place when it holds a disabled action that can explain itself. A disabled action with NO reason still earns nothing: that is a silent dead control, which the old rule was right about.
2. _button_for_action set display:none, width 0, height 0. Now mounted at height 1, disabled.
3. _widgets_for_action mounted the reason with display:none, height 0 and .console-hidden-control. Now a real row.

The reason stays its OWN row rather than being folded into the button label as DESIGN.md's short inert-action examples do ('Delete — built-in'). These reasons are full sentences -- 'Change tracking is off (git unavailable).' -- and at 33 columns a folded label would wrap to four rows of button.

Legible Disabled Rule: the override lives in the APP stylesheet, not widget DEFAULT_CSS, because Textual's Button:disabled ('text-style: bold dim' plus 'color: auto 50%') sits in the same tier and wins for a Button. text-style: none does not clear Textual's dim, so the colour carries it. The reason row is muted but explicitly NOT dimmed -- it is the only place several restrictions are explained at all.

Density cost, stated plainly: a rail with all three actions disabled now renders three headings + three buttons + three reasons where it previously rendered nothing. That is the intended trade -- the critique's 'wall of negatives' complaint is about content that says nothing, and these rows say why something is unavailable.

Modified: console_run_inspector.py, css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_run_inspector.py.
<!-- SECTION:NOTES:END -->
