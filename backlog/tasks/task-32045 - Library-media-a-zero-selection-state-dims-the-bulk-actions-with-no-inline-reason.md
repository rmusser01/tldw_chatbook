---
id: TASK-32045
title: >-
  Library media: a zero-selection state dims the bulk actions with no inline
  reason
status: To Do
assignee: []
created_date: '2026-09-08 14:37'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P2. In select mode with nothing selected, Export/Review/Delete are disabled with the '○' marker but no inline reason, while Analyze explains its block. This violates the surface's own 'explain why unavailable' rule that task-31981 established for the analysis actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With zero items selected, the disabled bulk actions carry a short inline reason (e.g. 'Select items to enable') the way the analysis block already does
- [ ] #2 Selecting an item clears the reason and enables the actions
- [ ] #3 A painted pin asserts the reason is present with zero selected and gone once an item is selected
<!-- AC:END -->
