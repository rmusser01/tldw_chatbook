---
id: TASK-31223
title: Footer - never advertise keys the focused widget will swallow
status: To Do
assignee: []
created_date: '2026-09-03 22:31'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P1: the footer shortcut branch keys off screen state, not focus; with focus in an Input it advertised '] next in set | m | R' while keystrokes were inserted as text (a stray ] corrupted the filter to a zero-match list). Keyboard-first brand: the footer is the instrument and must not lie.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With focus in a text input, the footer reflects typing context instead of advertising swallowed action keys
- [ ] #2 Walk keys shown in the footer always work when shown
<!-- AC:END -->


## Renumbering

Renumbered from task-31205 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).
