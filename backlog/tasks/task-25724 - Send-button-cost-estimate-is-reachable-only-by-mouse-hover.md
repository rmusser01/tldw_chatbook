---
id: TASK-25724
title: Send button cost estimate is reachable only by mouse hover
status: To Do
assignee: []
created_date: '2026-08-31 05:09'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Send label gains a dollar suffix when a cost estimate exists, but the estimate itself is delivered solely through a hover tooltip. In a keyboard-first terminal application hover is the one channel the primary audience does not use, so the information is unreachable. The unlabelled marker also changes the button's width on the keystroke path, shifting the composer's right edge while typing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cost estimate is reachable without a pointer
- [ ] #2 The cost affordance is labelled rather than represented by a bare symbol
- [ ] #3 Send button width does not change as the draft is typed
<!-- AC:END -->
