---
id: TASK-41637
title: "An MCP approval round with no running app waits forever"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- mcp
- agents
priority: medium
references:
- Tests/Chat/test_workflow_planning_native.py
dependencies:
- TASK-41314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** While retargeting the planning nodes, TASK-41314 saw that an MCP tool approval requested when no app is running, or once the app has gone, never settles. The round waits with no deadline and no refusal, so the run hangs instead of failing closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approval round with no running app settles as refused (fail closed) promptly
- [ ] #2 A test requests an approval with the app absent and asserts the refusal within a bounded time
<!-- AC:END -->
