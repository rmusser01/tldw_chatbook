---
id: TASK-23198
title: 'Console Context rail: fix the Tab focus trap and add rail keybindings'
status: To Do
assignee: []
created_date: '2026-08-29 21:56'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tab never leaves the Context rail: thirty consecutive presses stay inside, cycling nineteen stops. This is a WCAG 2.1.2 No Keyboard Trap failure. The rail also declares no BINDINGS, so there is no shortcut to toggle it, jump to a section, or collapse all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tab moves focus out of the Context rail rather than cycling within it
- [ ] #2 The rail exposes bindings to collapse all and expand all sections
- [ ] #3 A regression test walks Tab from inside the rail and asserts focus leaves
<!-- AC:END -->
