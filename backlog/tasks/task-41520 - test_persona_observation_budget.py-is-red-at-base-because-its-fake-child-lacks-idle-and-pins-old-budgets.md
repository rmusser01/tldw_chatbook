---
id: TASK-41520
title: "test_persona_observation_budget.py is red at base because its fake child lacks idle= and pins old budgets"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- personas
priority: medium
references:
- Tests/Backup_Recovery/test_persona_observation_budget.py
dependencies:
- TASK-41113
- TASK-41115
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the actor stream (round 15, TASK-41113 and TASK-41115). The test's fake `_child` does not accept `idle=` and the file pins the old budgets, so it is red at base after the observation-budget change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The fake child accepts the current signature and the budgets pinned are the current ones
- [ ] #2 The file passes at `-n 4` and is on a PR lane or the notes say why not
<!-- AC:END -->
