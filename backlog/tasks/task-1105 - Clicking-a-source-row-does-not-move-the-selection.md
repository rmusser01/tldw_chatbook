---
id: TASK-1105
title: >-
  Clicking a source row does not move the selection off the first row
status: Done
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
- [x] #1 Clicking a source row selects that row
- [x] #2 `Preview`, `Check now` and `Delete` act on the clicked source, not on row 0
- [x] #3 A test with **at least two** rows clicks the second and asserts it is selected, proven to fail against current code
- [x] #4 Keyboard cursor movement selects the same way
- [x] #5 The Runs, Items, Rules and Notifications tables are checked for the same defect
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause: the app's global focus outline erases the DataTable's own click metadata.**

`core/_reset.tcss` carries `*:focus { outline: solid $ds-focus-accent; }`. Textual paints an `outline` **over** the widget's outermost rendered lines rather than around them — that is what distinguishes it from `border`, which would cost geometry. For a `DataTable` the segments it overwrites lose the `{"row", "column"}` Rich style metadata, and `DataTable._on_click` resolves the clicked row from exactly that metadata:

```python
meta = event.style.meta
if "row" not in meta or "column" not in meta:
    return
```

so the bottom-most visible row of a **focused** table cannot be clicked at all. The first click is itself what focuses the table (`MouseDown` sets focus, then the `Click` is dispatched), so one click is enough to hit it. Measured in-harness, at the moment of the click:

```
MouseDown  style_after={'row': 1, 'column': 0}      <- correct
MouseUp    style_after={}                            <- table now focused
Click      style_after={}   -> cursor stays (0, 0)
```

Isolated in a bare Textual app to rule the app's own wiring out — same table, same clicks, CSS the only variable:

```
--- no-outline rows=6      --- outline rows=6
    click y=5 -> row 4         click y=5 -> row 4
    click y=6 -> row 5         click y=6 -> row 0   <- last row dead
```

The Sources table is three rows tall in the workbench (header + two rows), so its "last row" is row 1 — leaving row 0 as the only reachable row, which is the reported symptom.

**Fix.** `.watchlists-region DataTable:focus { outline: none; }` in `features/_watchlists.tcss`, plus a `> .datatable--cursor` rule using the sanctioned `$ds-focus-bg`/`$ds-focus-fg` pair so focus stays visible without repainting over content (Textual's own `DataTable:focus { background-tint }` also survives). Scoped to `.watchlists-region`, which every region on this screen carries.

**AC#5 — the other four tables had it too, and a second defect besides.** Runs, Items, Rules and Notifications only handled `RowSelected`/`CellSelected` (activation), so even with the cursor moving, a single click selected nothing. They now handle the highlight events like `SourcesPane`, gated by a new shared `table_selection.highlight_is_user_driven`: a `DataTable` also announces a highlight when it is *built*, and since these panes hold their rows in `recompose=True` reactives, treating that announcement as a user action made the pane fight its own screen — the runs deep link was overwritten by row 0, and `_apply_tree_scope` clearing a selection was undone by the rebuild the clearing caused. Both were caught by existing tests in `test_watchlists_destination_shell.py` when the gate was missing. Focus is the discriminator: a rebuilt table holds no focus, a clicked or arrowed one always does. `SourcesPane` deliberately does not use the gate — TASK-1100 relies on the row-0 highlight of a freshly populated sources table arming `Preview`/`Check now`.

`NotificationsPane` additionally seeds the new table's cursor from the surviving selection, because `selected_notification` is `recompose=True` and would otherwise bounce a clicked row back to row 0. `RunsPane`'s handlers are scoped to `#runs-table` so its `#runs-detail-items` sibling cannot clear the run selection that produced it.

**Verified in the running app** (clean scratch profile, two sources, `https://summitroute.com/blog/feed.xml`):

```
Selected: Krebs on Security      <- default, row 0
[click row 1]
Selected: Summit Route           <- the row that was clicked
[Check now]
Items: 10, every one Source = Summit Route   <- acted on the clicked source
```

**Out of scope, and real:** every other `DataTable` in the app still loses its bottom row to the same global outline. The rule here is deliberately scoped to Watchlists rather than applied to `components/_lists.tcss`, because a global change moves the focus affordance on every table-bearing screen and wants its own regression pass.

**Files:** `css/features/_watchlists.tcss` (+ regenerated `tldw_cli_modular.tcss`), `UI/Watchlists_Modules/table_selection.py` (new), `items_pane.py`, `runs_pane.py`, `rules_pane.py`, `notifications_pane.py`, `Tests/UI/test_watchlists_source_row_click_selects.py`.
<!-- SECTION:NOTES:END -->
