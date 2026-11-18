---
id: TASK-41636
title: "Workflow planning shutdown discards planning usage that was already received"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- accounting
priority: low
references:
- Tests/Chat/test_workflow_planning_native.py
dependencies:
- TASK-41314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41314 retargeted the native planning shutdown node: it now waits for the worker and accepts either current or corrected accounting. The current behaviour drops the usage of a planning call whose reply had already arrived when shutdown began, so spend is under-reported. The review asked that the test stop pinning the defect, and this task fixes it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Usage from a planning reply received before shutdown is recorded
- [ ] #2 The shutdown node in `Tests/Chat/test_workflow_planning_native.py` asserts the corrected accounting only
<!-- AC:END -->
