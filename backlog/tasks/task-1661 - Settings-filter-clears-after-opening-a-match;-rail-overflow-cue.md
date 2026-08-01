---
id: TASK-1661
title: 'Settings: filter clears after opening a match; rail overflow cue'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r3-p1
dependencies: []
priority: high
---

## Description (the why)

Critique round 3 P1: after Enter opened a filtered match the query remained, pruning the rail to the last search's matches for the rest of the session with no advertised clear affordance; separately the unfiltered rail hid four below-fold categories behind an invisible scrollbar.

## Acceptance Criteria (the what)

- [x] Enter-open clears the query and restores the full rail
- [x] The rail scrollbar is visible against the panel (overflow cue)

## Implementation Notes

`_submit_category_search` clears query+input+filter after `_select_category`; rail scrollbar-color bumped $ds-grid-line -> $ds-text-muted. Live-verified: 'No filter' + full rail after opening Console Behavior.
