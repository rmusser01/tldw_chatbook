---
id: TASK-32327
title: >-
  Responsive rail force-collapse posts a visible notice
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B3. Width-band rules silently force-collapse rails (<150 Inspector, <100 Context, <84 single-pane) and resolve both-open conflicts (console_rail_state.py resolve_console_rail_priority). Users see their rail disappear with no explanation. Post a transient notice the first time per session a responsive override hides a rail the user had open.

Filed from the 2026-09-10 Console rail UX review (review item B3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a responsive width rule closes a rail the user had open, a transient notice names what happened and how to reopen (once per session per rail, not per resize tick)
- [ ] #2 Resize spam does not produce repeated notifications (debounced/once-per-session)
- [ ] #3 Preferences are still not rewritten by responsive overrides (existing behavior kept)
- [ ] #4 Unit tests cover the notice firing once and not on subsequent collapses
<!-- AC:END -->
