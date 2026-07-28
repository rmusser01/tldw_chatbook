---
id: TASK-1140
title: >-
  The global focus outline makes the bottom row of every DataTable unclickable
status: To Do
assignee: []
created_date: '2026-07-28 12:00'
labels:
  - bug
  - ui
  - css
  - a11y
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`*:focus { outline: solid $ds-focus-accent }` in `css/core/_reset.tcss` paints Textual's focus outline **over** a widget's outermost lines. Those segments lose the `{"row", "column"}` meta that `DataTable._on_click` reads to resolve which cell was hit — so a click on the bottom row of any focused table does nothing.

Isolated in a bare Textual app during the TASK-1105 fix: a six-row table with the outline leaves the cursor at row 0 when the last row is clicked, and lands correctly on row 5 without it. It is masked on first interaction because `MouseDown` focuses the table before the `Click` is resolved, so the very first click still works — which is why it reads as intermittent rather than broken.

TASK-1105 fixed this for the Watchlists tables. **Every other `DataTable` in the app still has it**, and the app has many.

The fix likely belongs in `components/_lists.tcss` — moving the focus affordance off the outermost line so it stops consuming the click target. That relocates a focus indicator app-wide, which is why it was not folded into 1105: it wants its own regression pass across every screen that shows a table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The bottom row of a focused `DataTable` is clickable
- [ ] #2 Focused widgets still show a visible focus affordance
- [ ] #3 A test clicks the last row of a focused table and asserts the cursor moved, proven to fail against current code
- [ ] #4 Screens with tables outside Watchlists are checked, and the ones affected listed here
<!-- AC:END -->
