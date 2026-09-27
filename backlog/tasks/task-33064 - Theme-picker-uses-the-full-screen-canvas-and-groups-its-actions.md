---
id: TASK-33064
title: Theme picker uses the full-screen canvas and groups its actions
status: To Do
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. At 211x44 and 235x52 the theme list is capped at 22 rows (of ~95) leaving 14-19 rows empty, and the card's 9-12 actions are one ungrouped full-width stack with Revert in the middle of the file actions. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52 the list fills the available height (no empty band below it)
- [ ] #2 Switching actions (Use, Try, Revert), creation actions (Clone, New, Import) and your-theme actions (Edit, Rename, Export, Delete) are visibly grouped, laid out horizontally when the card is wide enough
- [ ] #3 At 80x24 every control stays reachable and the list shows at least 5 rows
<!-- AC:END -->
