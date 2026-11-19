---
id: TASK-41646
title: "The workflow driver's failure handler leaves never-launched preparations open for reconcile"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41320 made reconcile close a retained run's never-launched, unwaved preparation, so the chat accepts a new turn. The driver's own failure handler sees that same preparation first but leaves it open, so the run stays retained until a reconcile happens to run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The driver handler closes a never-launched preparation itself, recording the same review row reconcile records
- [ ] #2 A test fails a run before launch and asserts that the chat accepts a new turn without reconcile
<!-- AC:END -->
