---
id: TASK-21142
title: 'Wizard keyboard model: Enter, focus order, radio selection'
status: To Do
assignee: []
created_date: '2026-08-25 06:14'
labels:
  - ux
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings N-1, N-2, N-8, N-9 (findings.md section N): Enter never advances; the abandon action is the first Tab stop after step content; radio highlight is not selection so Down+Next silently keeps the Quick track; Back does not restore focus to the step's primary control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Enter advances the wizard when the focused widget does not consume it; Input submit advances; track radio Enter advances
- [ ] #2 Track-choice selection follows the highlight (Down then Next yields the Full track)
- [ ] #3 Tab from step content reaches Next before any abandon action
- [ ] #4 After Back, focus lands on the step's primary control with a visible indicator
- [ ] #5 Existing wizard tests pass
<!-- AC:END -->
