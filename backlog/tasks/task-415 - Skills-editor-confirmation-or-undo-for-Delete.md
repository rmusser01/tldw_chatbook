---
id: TASK-415
title: Skills editor - confirmation or undo for Delete
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). Delete in the skill editor is a single click with no confirmation and no undo and it sits directly beside Save with equal visual weight at the bottom of a scrolled form. A misclick permanently removes the skill directory including supporting files. Inconsistent with how carefully the same panel gates trusting a skill. NNG heuristic 5 (error prevention).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deleting a skill requires an explicit confirmation step or offers a working undo,Confirmation communicates what will be removed including supporting files count when present,Save and Delete are visually differentiated (destructive styling or separation),Covered by tests
<!-- AC:END -->
