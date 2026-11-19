---
id: TASK-41957
title: "Change Review initialization tests red on dev: the registry source-change node and five slow readiness waits"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- workspaces
priority: low
references:
- Tests/Workspaces/test_native_change_review_initialization.py
dependencies:
- TASK-41635
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** `test_admitted_callback_cannot_publish_ready_after_original_source_changes[registry]` ends with `executions != ()` on dev and on TASK-41635's branch; five source-replacement and readiness-handoff waits in the same file also fail on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node has a recorded cause and passes after repair or fix
<!-- AC:END -->
