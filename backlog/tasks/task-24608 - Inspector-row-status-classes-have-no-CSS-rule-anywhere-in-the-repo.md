---
id: TASK-24608
title: Inspector row status classes have no CSS rule anywhere in the repo
status: To Do
assignee: []
created_date: '2026-08-30 00:54'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every run-inspector row is built with a console-inspector-row-<status> class and the modifier is swapped on every in-place update, but grep across all stylesheets returns zero matches for console-inspector-row, including the base class. For the Provider row this is harmless because the status word is the value. For Sources and Approvals it is not: Approvals carries status blocked when the pending count is above zero while its text reads only 'N pending', so a pending approval renders identically to none pending. The class has been attached, swapped and covered by passing tests for as long as the rule has been missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Blocked, running and ready inspector rows are visually distinguishable using semantic status tokens
- [ ] #2 Colour is reinforcement only; every blocked row still reads as blocked from its text alone
- [ ] #3 A repo check fails when a class attached in Python has no matching rule in the stylesheet bundle
<!-- AC:END -->
