---
id: TASK-31951
title: >-
  Library - opt the Conversations, Skills and Collections readers into the
  1-cell pane grip
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H Task 2 ruling: PR H made LibraryAdaptiveReaderPaneGrip's width per reader profile so Media could drop from five columns to one, which is the defect task-31633 fixed for Media. Conversations, Skills and Collections still carry the five-column grip - ten dead columns per surface - because H changed only the profile Media uses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three sibling readers render a 1-cell grip with the same glyphs as Media
- [ ] #2 Their resolver width pins are updated to the new reservation
- [ ] #3 No reader surface reserves grip columns it does not paint
<!-- AC:END -->
