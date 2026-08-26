---
id: TASK-15201
title: 'Fleet panel: no View all tail and no auto-scroll to the expanded section'
status: To Do
assignee: []
created_date: '2026-08-11 04:01'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Disclosed scope cut from supervisor-fleet PR 2b Task 4. The plan's state-2 description named 'Scrollable/virtualized past a screenful; View all opens full run history'; what shipped relies on the rail's existing outer VerticalScroll and has no View all tail. The reviewer probed this properly rather than assuming: with 12 live rows at 180x48, the 12th row's unclipped region sits inside the viewport but the compositor hit-test resolves to another widget — i.e. not painted by default — while after an explicit scroll_visible() the row IS painted at its own region and a real click routes to the right child. So nothing is permanently unreachable and routing is scroll-position independent; this is NOT a task-226 clipping bug. It is a real UX gap: the fleet section sits ~5th among 6-7 peer sections in one shared scroll, nothing auto-scrolls a newly-expanded section into view, and with a dozen live children the user must scroll past every other section to reach the bottom rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Expanding the fleet section scrolls it into view
- [ ] #2 A View all affordance opens full run history when rows exceed what the rail can show
- [ ] #3 A row past the fold is reachable and clickable without manual hunting (verified with a compositor hit-test, not DOM presence)
<!-- AC:END -->
