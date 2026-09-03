---
id: TASK-30045
title: Review sets - set identity and per-item state in the Reader chrome
status: To Do
assignee: []
created_date: '2026-09-03 13:06'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique 2026-09-03 P2 + user ruling on Q2: the review set is a workflow object and deserves a real runtime surface, not only a footer string. Today the set's name never appears while walking and the current item's reviewed state is displayed nowhere (m's only feedback is the aggregate counter changing).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While a set is active the Reader shows the set's name and progress in its chrome
- [ ] #2 The current item's reviewed state is visible at a glance and updates when m toggles it
- [ ] #3 The chrome disappears when no set is active
<!-- AC:END -->
