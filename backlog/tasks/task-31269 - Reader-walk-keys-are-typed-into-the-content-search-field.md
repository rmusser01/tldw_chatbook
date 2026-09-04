---
id: TASK-31269
title: Reader walk keys are typed into the content search field
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P0 (my #2367 regression): the content search bar's on_mount focuses its Input whenever the query is empty, and the Analysis tab (task-28026) mounts that bar whenever an analysis exists. So in Analysis mode every ]/[ item load moves focus into the box and the next key is typed as text (A cap 22/24/36: `▊ ]`); in Read mode the same happens whenever Find is open across a walk (B cap_32b `]]]]]`, cap_41-44 with the review banner frozen). The Find button does not toggle the bar off and Escape from inside the input only blurs on the first press. Workflow 3 (review every analysis with ]) is exactly this path and fails silently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In Analysis mode, walking two items with ] never leaves focus in the search Input (test asserts `screen.focused` is not the Input after each load)
- [ ] #2 With Find open in Read mode, ] [ m R still act as keys across item changes — the bar keeps its query but never re-takes focus on an item change
- [ ] #3 The Analysis-tab search bar is collapsed until Find is invoked, matching the Read tab
- [ ] #4 Pressing Find while the bar is open closes it; Escape from inside the Find input closes the bar in one press on every tab
- [ ] #5 Live-verified in tmux: an Analysis-mode walk over three items using only ]
<!-- AC:END -->
