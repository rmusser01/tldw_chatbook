---
id: TASK-287
title: Pin footer CSS/recompose drift guards with tests
status: To Do
assignee: []
created_date: '2026-07-17 15:17'
labels:
  - ux
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-264's final review left two comment-only invariants unguarded: (1) AppFooterStatus.DEFAULT_CSS must stay a faithful subset of the live bundle source (css/components/_widgets.tcss 'Window Footer Widget' block) — drift silently diverges harness geometry from production; (2) PersonasScreen's footer hints use the non-persisting set_shortcut_context path, which is only safe while PersonasScreen has no recompose path — a future recompose=True reactive would silently drop its hints. Make both a red test instead of a comment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test fails when AppFooterStatus.DEFAULT_CSS core rules diverge from the _widgets.tcss footer block
- [ ] #2 A test fails if PersonasScreen gains a screen-level recompose path while still using the non-persisting footer registration
<!-- AC:END -->
