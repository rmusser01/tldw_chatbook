---
id: TASK-32949
title: Leaving Settings with unsaved theme edits drops them silently
status: To Do
assignee: []
created_date: '2026-09-24 22:30'
labels:
  - settings
  - theme
  - ux
priority: medium
dependencies:
  - TASK-32941
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32941 added a Stay / Discard / Save prompt when leaving the Theme category with unsaved edits, but only for category switches inside Settings. Navigating away from the Settings screen itself (tab bar, command palette, Ctrl+digit routes) still discards the in-progress theme silently, exactly the data loss 32941 closed for the in-screen path. Found by the 2026-09-24 branch review of `fix/theme-ux-wave`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving the Settings screen by any navigation route while the theme editor has unsaved edits shows the same Stay / Discard / Save choice as a category switch
- [ ] #2 Stay keeps the user on Settings ▸ Theme with the edit intact; Discard leaves and drops it; Save saves then leaves (a refused save or pending overwrite keeps the user on Theme)
- [ ] #3 Navigation away from Settings with no unsaved theme edits is unchanged (no prompt, no added latency)
- [ ] #4 Pilot tests cover each route used and each choice
<!-- AC:END -->
