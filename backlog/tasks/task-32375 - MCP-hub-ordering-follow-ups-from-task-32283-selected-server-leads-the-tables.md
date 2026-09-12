---
id: TASK-32375
title: MCP hub ordering follow-ups from task-32283 (selected server leads the tables)
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - mcp
  - tests
dependencies: []
priority: low
---

## Description

Deferred minors from task-32283: the workbench→canvas `selected_server_key` pass-through is not asserted directly; the permissions ordering test bypasses `_select_server_key()`; the tool-catalog drill degrades silently to the unfiltered list when the catalog is empty; Tools order keys off the rail selection even when the filter says "All servers" (allowed by ruling R8).

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] A workbench-level test asserts the selected server key reaches both canvases
- [ ] The permissions ordering test goes through `_select_server_key()`
- [ ] An empty-catalog drill shows an explicit empty state rather than the unfiltered list
