---
id: TASK-31264
title: Themed light modes expose hard-coded slate colors in stock widget states
status: To Do
assignee: []
created_date: '2026-09-04 13:47'
labels:
  - ui
  - themes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under the new light themes from PR #2374 (observed live under apricot at 2516735cfd), some widget states paint a slate blue-gray that no theme variable controls: the model-catalog consent dialog's non-primary 'Don't check' button renders bg #4f6379 / fg #e8eaed, and the Library rail's selected row (Skills) shows a similar slate highlight. Traced far enough to rule out the obvious suspects: the dialog's own DEFAULT_CSS uses $panel/$background correctly, no app .tcss contains these hexes, and they do not derive from the active theme's palette via Textual's ColorSystem — so the origin is likely Textual stock component styles or a widget-tier default the app never overrides (root cause unconfirmed; note the widget-tier-CSS-loses-to-app-tier lesson). Cosmetic on dark themes, clearly foreign on light ones.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The source of the slate paint is identified and recorded in this task
- [ ] #2 Non-primary dialog buttons and rail selected rows follow the active theme's palette under a light theme (verified live under apricot)
<!-- AC:END -->
