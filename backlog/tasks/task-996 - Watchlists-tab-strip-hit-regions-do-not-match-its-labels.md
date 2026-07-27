---
id: TASK-996
title: >-
  Watchlists tab strip activates the wrong section for a given click column
status: To Do
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking the column where `Items` is drawn in the section tab strip activates `Runs` instead. Reproduced repeatedly at 235x52 on `origin/dev` `dbbb7de84`: computing the click column from `index($0, "Items")` and from `index($0, "    Items")+6` both landed on `Runs`, and `Items` was never reached by mouse.

So the tabs are mislabelled from the user's point of view — the thing you click is not the thing you get.

Very likely the same root as task-875's fix: `WatchlistsTabStrip` pins itself to `height: 1` while its `Button`s want three rows, so their layout boxes — and therefore their hit regions — do not line up with where their labels are painted. The label fix in `features/_watchlists.tcss` made the text visible; it may not have corrected the geometry.

The dialog `Create` button showing the same symptom (see `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`) suggests checking whether this is broader than the tab strip.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking any tab activates the section whose label is under the pointer
- [ ] #2 A test drives a real click at each tab's rendered label column and asserts the resulting `active_section`, proven to fail against current code
- [ ] #3 The active tab stays visually distinguishable and the strip stays one row
- [ ] #4 It is established whether the dialog `Create` button shares this cause, and the answer recorded here
<!-- AC:END -->
