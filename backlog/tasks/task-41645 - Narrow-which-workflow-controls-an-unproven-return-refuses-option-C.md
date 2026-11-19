---
id: TASK-41645
title: "Narrow which workflow controls an unproven return refuses (option C)"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- console
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- backlog/decisions
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41320 made an unproven control return or metadata read refuse further controls with `workflow_return_unproven`, and the Console says that a restart releases held chats. On 2026-11-17 the owner chose restart-only and asked for the narrowing to be filed. Under option C, only controls that act on the unproven resource refuse; the rest keep working without a restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-285 records which controls an unproven return refuses and why
- [ ] #2 Controls outside that set work on a run with an unproven return
- [ ] #3 Tests cover one refused and one allowed control
<!-- AC:END -->
