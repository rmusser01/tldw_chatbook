---
id: TASK-1362
title: A small edit to a long page never clears the change threshold
status: To Do
assignee: []
created_date: '2026-07-29 23:55'
labels:
  - watchlists
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`change_threshold` defaults to `0.1` and is compared against `calculate_change_percentage`, which is
whole-page character similarity via `difflib.SequenceMatcher`. On a long page, a genuinely important
small edit — a price, a version number, a single added paragraph — moves that ratio by far less than
0.1, so **no item is ever created and the user is never told**.

The failure is silent and indistinguishable from "nothing changed", which is the same class of
problem as the watchlists that never checked at all (TASK-1210): the machinery works, and the user
concludes the feature does nothing.

Found while implementing TASK-1343, which made the change body renderable and therefore made the
threshold's behaviour visible for the first time.

Worth considering together: a per-region or per-element comparison, an absolute floor alongside the
ratio (e.g. "N characters changed"), or surfacing the computed percentage in the UI so a user can see
why nothing fired and tune the threshold. `baseline_manager.py` already contains structural and
key-element comparison that would help here, but it is orphaned — see TASK-1360.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A small but meaningful edit to a long page produces an item under the default configuration
- [ ] #2 The rule that decides "significant" is stated in the UI or docs, so a user can tell why a change did or did not fire
- [ ] #3 A test pins a realistic long-page-small-edit case and fails under the old whole-page ratio alone
<!-- AC:END -->
