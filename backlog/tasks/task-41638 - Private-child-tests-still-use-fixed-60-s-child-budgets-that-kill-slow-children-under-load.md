---
id: TASK-41638
title: "Private-child tests still use fixed 60 s child budgets that kill slow children under load"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- flaky
priority: low
references:
- Tests/Chat/test_workflow_planning_native.py
dependencies:
- TASK-41314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41314 raised its private-child budgets to 180 s because loaded runners killed children that were still making progress. Other tests that run a private child keep a fixed 60 s budget, so they fail the same way under load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every private-child test either uses a budget that tolerates load or uses the shared CPU-idle-aware wait
- [ ] #2 A grep for a fixed 60 s child budget in `Tests/` finds no remaining site, or each remaining one is justified in a comment
<!-- AC:END -->
