---
id: TASK-865
title: >-
  The watchlist tree does not show which node the screen is scoped to
status: To Do
assignee: []
created_date: '2026-07-26 23:20'
labels:
  - watchlists
  - ui
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`WatchlistTree` renders its nodes as buttons and never reads `tree_scope`, so nothing in the left rail indicates which node the centre is currently scoped to. The Feeds region's heading (`Feeds in Morning AI Brief (1)`) is the only feedback the user gets.

That was tolerable while the scope only drove a breadcrumb. Phase C made it drive what the Feeds region renders and what "Stage in Console" sends, so the tree is now the primary navigation control for the screen — and it gives no sign of its own state.

Two related gaps found in the same review, worth folding in:

- The panes have no selected-row styling either. `SourcesPane`, `RunsPane` and `NotificationsPane` rely entirely on Textual's stock DataTable focus cursor, which is a focus affordance rather than a selection indicator — it always sits somewhere, including on rows the screen does not consider selected. Phase C deliberately declined to move it, because relocating it to row 0 on a tree move would assert a different wrong selection rather than fix one.
- `_load_tree_data` logs at debug where every sibling loader on the screen calls `notify(severity="error")`. A real database failure therefore renders identically to "you have zero watchlists" — two empty roots and no message. The tree is its own only error surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The tree node matching the current `tree_scope` is visually distinguished from its siblings
- [ ] #2 The highlight follows a breadcrumb promotion as well as a direct tree click, since both move the scope through `_apply_tree_scope`
- [ ] #3 The highlight survives a section switch and a rail toggle, both of which rebuild the tree from screen-held state
- [ ] #4 Selected rows in the Sources, Runs and Notifications panes are distinguishable from merely focused ones
- [ ] #5 A failure inside `_load_tree_data` is surfaced to the user distinguishably from an empty result
- [ ] #6 Tests pin each highlight against the production stylesheet, not a bare `App`
<!-- AC:END -->
