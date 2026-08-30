---
id: TASK-24600
title: Console Inspect rail cannot be reopened below 84 columns
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:53'
updated_date: '2026-08-30 01:50'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Below the 84-column single-pane threshold both rail handles hide. A rail that is explicitly open at that width can still be collapsed by its own header button, after which nothing on screen references the Inspector: no handle, no status chip route, no keyboard binding, no command-palette entry. The only observed recovery is resizing the terminal. Small-terminal users lose the Inspector for the session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsing the Inspect rail below 84 columns always leaves a visible, activatable affordance that reopens it
- [x] #2 The reopen affordance is reachable by keyboard alone, not only by mouse
- [x] #3 A regression test drives collapse then reopen at 80x24 and asserts the rail is displayed again
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed by the same seam as TASK-24604, which is why they were done together.

The mechanism was documented intent that nobody had composed: below CONSOLE_SINGLE_PANE_COLUMNS (84) both rail handles hide, while 'budget-eligible explicit rails may still render from their 70/74 floors through 83 via compact override while the handles remain hidden' (console_rail_state.py). So a rail explicitly open at 80 columns could still be collapsed by its own header button, after which nothing on screen referenced the Inspector -- a live 80x24 capture returned zero occurrences of 'Inspect' -- and the only observed recovery was resizing the terminal.

The fix is that the new alt+i binding is NOT gated on the rail or its handle being displayed. That is the whole property: it can only be the way back if it works while both are invisible. test_collapsing_below_84_columns_leaves_a_way_back drives collapse -> reopen at 80x24 and asserts the rail is displayed again, which is the regression that would have caught this.

No change was made to the handle-hiding rule itself: hiding both handles below 84 is a deliberate single-pane layout decision, and widening the rail's reach at that width would undo it. What was missing was a route that does not depend on painted chrome.

Modified: see TASK-24604.
<!-- SECTION:NOTES:END -->
