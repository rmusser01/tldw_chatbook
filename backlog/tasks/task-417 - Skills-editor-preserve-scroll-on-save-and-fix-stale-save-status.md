---
id: TASK-417
title: Skills editor - preserve scroll on save and fix stale save status
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). Pressing Save at the bottom of the editor snaps the viewport back to the top so the 'Saved.' status line and the trust-state change render below the fold unseen. The 'Saved.' text then persists indefinitely - it still read 'Saved.' after a later bootstrap-trust action. NNG heuristic 1 (visibility of system status).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Save the viewport still shows the Save button and adjacent status line (scroll position preserved or scrolled to the status),Save status is cleared or replaced when a different action runs so it never reports a stale outcome,Save outcome is perceivable without scrolling,Covered by tests
<!-- AC:END -->
