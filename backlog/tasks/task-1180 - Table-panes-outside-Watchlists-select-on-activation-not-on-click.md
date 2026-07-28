---
id: TASK-1180
title: >-
  Table panes outside Watchlists select on activation, so a click moves the
  cursor but selects nothing
status: In Progress
assignee: []
created_date: '2026-07-28 14:00'
labels:
  - bug
  - ui
  - a11y
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking a row in the MCP Permissions and MCP Tools tables moves the DataTable cursor but leaves the Inspector empty — nothing is actually selected.

Those panes handle `DataTable.RowSelected`, which Textual fires on **activation** (Enter, or a second click), not `RowHighlighted`, which is what a single click produces. So a click highlights a row and selects nothing.

This is the same defect TASK-1105 fixed for the Watchlists Sources pane, where it made `Preview` / `Check now` / `Delete` unreachable by mouse and ultimately hid a dead scrape path (TASK-1100). Watchlists was fixed in isolation; **every other table pane in the app still has it.**

Found while verifying TASK-1160's app-wide focus-outline fix, which made the bottom row clickable everywhere — that exposed this second layer, since a click now reaches the right row and still does nothing.

Note the two are independent: TASK-1160 was about clicks not *landing*, this is about a landed click not *selecting*. Both had to be true for a mouse user to get anywhere.

23 files under `tldw_chatbook/UI` use `DataTable`; the two confirmed live are on the MCP screen, and the rest should be audited rather than assumed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking a row in the MCP Permissions and MCP Tools tables selects it, and the Inspector reflects the selection
- [x] #2 Every table pane under `tldw_chatbook/UI` is audited, and the affected ones listed here or fixed
- [x] #3 Keyboard cursor movement selects the same way as a click
- [x] #4 A shared mechanism is preferred over per-pane handlers, so the next table added does not need to remember
- [x] #5 A test clicks a row and asserts the selection, proven to fail against current code
<!-- AC:END -->

## Implementation Notes

`DataTableClickSelectMixin` (`UI/Widgets/table_click_select.py`) forwards user-driven cursor
movement to a pane's existing `on_data_table_row_selected`, so each pane keeps its own row-key
resolution and only declares that a click should reach it.

**Audit (AC#2).** An AST pass over the 14 panes under `UI/` that handle either event found **7**
selecting on activation only: `mcp_audit_mode`, `mcp_permissions_mode`, `mcp_servers_mode`,
`mcp_tools_mode`, `Voice_Cloning_Window`, `stts_profile_library`, and — a **false positive** —
`mcp_workbench`, whose handlers take the panes' own custom messages rather than `DataTable` events,
so it works once the panes post. The five Watchlists panes were already fixed by TASK-1105;
`Evals/results_grid` and `schedules_workbench` correctly use highlight-only.

`stts_profile_library` cannot use the mixin: its handler is bound by `@on(DataTable.RowSelected,
"#stts-profile-table")`, and the mixin dispatches by conventional method name. It gets an explicit
paired handler with the same focus gate spelled out.

**The naive version of this fix is a feedback loop, and I shipped it into my own branch first.**
`RowHighlighted` is not a user-input event — a pane rebuilding its table emits it too, because
`clear()` resets the cursor to row 0. On the MCP workbench a selection triggers an awaited
remove/mount in `MCPInspector`, which re-syncs the mode, which repopulates the table, which
highlights row 0 again. Opening the Tools tab with **zero user input produced 157 selections**, and
buried a genuine click under the repeats. Three layers now prevent it, each earning its place from
an observed failure:

- **Focus gate.** Measured on the real screen: a repopulating table is not focused, while a click
  focuses it before the cursor moves and keyboard navigation requires focus by definition.
- **`repopulating_table()`**, declared by the pane before `clear()`/`add_row()`. Focus alone was not
  enough — switching the selected server rebuilds the tools table *while it still has focus*, and
  the row-0 transient then re-selected a tool and defeated the clear that triggered the rebuild.
  The existing `test_switching_selected_server_clears_tool_detail` caught that. Suppression releases
  after the next refresh rather than at the end of a `with` block, because Textual delivers a
  rebuild's highlight messages *after* the producing code returns; the obvious context-manager shape
  would have looked right and covered nothing.
- **Dedup** on the last forwarded row key, bounding the damage if a future pane forgets to declare.

**Two panes already knew about this hazard.** `mcp_permissions_mode` and `mcp_audit_mode` carry
comments explaining that `DataTable.clear()` resets `cursor_coordinate` to (0, 0), and they restore
the cursor by row key afterwards. The first version of this fix would have silently defeated that
careful existing work by treating the transient as a selection. Reading their comments turned the
fix from "add a guard" into "declare the rebuild so the existing restore still wins."

**The Watchlists panes do not have the storm** (measured: 0 spurious calls). `select_source_by_id`
sets local state without the inspector-remount feedback cycle, so their hand-written handlers are
left alone rather than churned. That does leave two patterns in the tree; the mixin is the one to
use for new tables.

**Mutation testing corrected itself.** The first check passed with the focus gate removed, because
the dedup guard independently suppressed the storm — a test that survives removing either guard
proves neither. Isolating gave 0 / 0 / **162** (focus-only / dedup-only / neither), and each guard
now has a test that fails when exactly that guard is removed.

**Baseline.** Identical command on `6577c67cf` and on this branch: the same **7** failures both
sides — 4 `PrivatePathError: link_or_non_regular` (environment) and 3 audit-redaction, all
pre-existing. `test_switching_selected_server_clears_tool_detail` passes on baseline, failed on the
first draft, and passes now.

Added: `tldw_chatbook/UI/Widgets/table_click_select.py`,
`Tests/UI/test_table_click_selects.py`, `Tests/UI/test_mcp_table_click_selects_end_to_end.py`.
Modified: the four `MCP_Modules` panes, `Voice_Cloning_Window.py`, `stts_profile_library.py`.
