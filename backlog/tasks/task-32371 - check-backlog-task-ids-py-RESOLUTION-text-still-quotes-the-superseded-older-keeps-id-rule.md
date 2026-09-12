---
id: TASK-32371
title: check_backlog_task_ids.py RESOLUTION text still quotes the superseded older-keeps-id rule
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - tooling
  - backlog
dependencies: []
priority: low
---

## Description

`backlog/docs/lessons-backlog-hygiene.md` (2026-09-08) records that a landed id beats an older-but-unlanded one, and this wave renumbered its own tasks 32272-32276 → 32341-32345 on that basis, but the guard script's RESOLUTION guidance still tells the reader that the older arrival keeps the id.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] The RESOLUTION text in `scripts/check_backlog_task_ids.py` states the landed-keeps-id rule and points at the lessons entry
- [ ] The guard's own test (if any) asserts the new wording
