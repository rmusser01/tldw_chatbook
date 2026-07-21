---
id: TASK-423
title: Skills discoverability - palette command deep link to Library Skills
status: To Do
assignee: []
created_date: '2026-07-21 15:19'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review (verified live). Skills has no nav tab (by design) and no command palette hit: typing 'skills' into Ctrl+P surfaces only a fuzzy 'Switch to Library' match. The only path is knowing to look inside Library's Browse rail. The legacy 'skills' route already resolves to Library so plumbing exists; there is no palette entry or deep link to the Skills rail row. NNG heuristic 6 (recognition rather than recall).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A command palette entry matching 'skills' navigates to Library with the Skills rail row selected,Legacy 'skills' route deep-links land on the Skills row rather than generic Library,Covered by tests
<!-- AC:END -->
