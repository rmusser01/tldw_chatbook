---
id: TASK-33241
title: Returning to the theme picker after Save repaints faster
status: To Do
assignee: []
created_date: '2026-09-28 08:00'
labels:
  - settings
  - theme
  - perf
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #2877's review: Save and Save as return to the picker, and that view switch (like Back) costs ~170 ms of restyle and render at 211x44 — not stylesheet parsing (TASK-33120 already removed that).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dominant cost of the editor-to-picker switch is measured and identified
- [ ] #2 The switch is cut to under 100 ms at 211x44, or a documented reason why not and the best achievable number
<!-- AC:END -->
