---
id: TASK-1105
title: >-
  Clicking a source row does not move the selection off the first row
status: To Do
assignee: []
created_date: '2026-07-28 09:00'
labels:
  - watchlists
  - bug
  - ui
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the Sources table, clicking any row leaves the selection on row 0. Measured with two sources in a table tall enough to show both (`Region(height=3)`, `row_count: 2`):

```
off (4, 1) -> local:subscription:1
off (4, 2) -> local:subscription:1
off (4, 3) -> local:subscription:1
```

The DataTable cursor never moves, so `selected_source` is always the first row. A user with several sources can only ever act on the first one — `Preview`, `Check now` and `Delete` all target row 0 regardless of what was clicked.

TASK-1100 added `on_data_table_row_highlighted`/`on_data_table_cell_highlighted`, which made the *default* highlight of row 0 select a source. That is what unblocked fetching, and it is real — but it is not click-to-select, and the first version of that PR's test claimed otherwise. With a single source in the fixture the assertion passed on the default selection alone; Qodo caught it, and widening the fixture to two rows made the true behaviour visible.

The likely cause is that the click reaches the table but does not move its cursor — check whether the table takes focus on click, and whether `cursor_type` and the mouse handling agree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking a source row selects that row
- [ ] #2 `Preview`, `Check now` and `Delete` act on the clicked source, not on row 0
- [ ] #3 A test with **at least two** rows clicks the second and asserts it is selected, proven to fail against current code
- [ ] #4 Keyboard cursor movement selects the same way
- [ ] #5 The Runs, Items, Rules and Notifications tables are checked for the same defect
<!-- AC:END -->
