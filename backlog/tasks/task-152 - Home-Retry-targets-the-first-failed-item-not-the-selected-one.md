---
id: TASK-152
title: 'Home Retry targets the first failed item, not the selected one'
status: To Do
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - follow-up
  - home
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Home's Retry control dispatches through the unscoped summarize_home_dashboard controls (home_screen.py:~363) while the rendered canvas buttons come from the selection-scoped build_home_controls. With 2+ retryable failed items, selecting a non-first item and pressing Retry requeues the FIRST failed item instead. Surfaced during F1b review; pre-existing dual-control-list seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing Retry on a selected failed Home item requeues THAT item
- [ ] #2 Regression test with 2+ retryable failed items asserts the selected item is the one requeued
<!-- AC:END -->
