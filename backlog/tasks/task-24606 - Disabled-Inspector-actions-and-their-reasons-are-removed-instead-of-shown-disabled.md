---
id: TASK-24606
title: >-
  Disabled Inspector actions and their reasons are removed instead of shown
  disabled
status: To Do
assignee: []
created_date: '2026-08-30 00:54'
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
- [ ] #1 A disabled Inspector action stays mounted and visibly disabled rather than being removed from layout
- [ ] #2 The reason an action is unavailable is readable without a mouse, for example carried in the label
- [ ] #3 A disabled action label meets at least 3:1 against its own background, measured in a running terminal
- [ ] #4 A test asserts the disabled action and its reason are present and displayed
<!-- AC:END -->
