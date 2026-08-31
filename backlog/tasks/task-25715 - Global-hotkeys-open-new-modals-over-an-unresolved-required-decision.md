---
id: TASK-25715
title: Global hotkeys open new modals over an unresolved required decision
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a blocking interrupt card is mounted and asking the user to choose one action, Ctrl+K and F1 each open another modal on top of it, producing three stacked layers with the original decision still pending. Panels also render without a scrim, so a modal reads as an inline card and clicks outside it silently do nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Global hotkeys that open panels are suppressed while a blocking decision card is mounted
- [ ] #2 Modal surfaces render a scrim that visually separates them from live content
- [ ] #3 Clicks outside a modal produce a visible response rather than silently doing nothing
<!-- AC:END -->
