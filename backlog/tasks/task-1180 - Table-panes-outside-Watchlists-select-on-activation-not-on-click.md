---
id: TASK-1180
title: >-
  Table panes outside Watchlists select on activation, so a click moves the
  cursor but selects nothing
status: To Do
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
- [ ] #1 Clicking a row in the MCP Permissions and MCP Tools tables selects it, and the Inspector reflects the selection
- [ ] #2 Every table pane under `tldw_chatbook/UI` is audited, and the affected ones listed here or fixed
- [ ] #3 Keyboard cursor movement selects the same way as a click
- [ ] #4 A shared mechanism is preferred over per-pane handlers, so the next table added does not need to remember
- [ ] #5 A test clicks a row and asserts the selection, proven to fail against current code
<!-- AC:END -->
