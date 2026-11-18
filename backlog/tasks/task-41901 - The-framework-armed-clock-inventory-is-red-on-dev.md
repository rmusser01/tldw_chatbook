---
id: TASK-41901
title: "The framework-armed clock inventory is red on dev"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
- architecture
priority: high
references:
- Tests/Architecture/test_framework_armed_clock_inventory.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-16 coordinator (2026-11-18): `test_no_unclassified_auto_refresh_arming` fails and `test_every_framework_armed_clock_is_classified` / `test_inventory_rows_are_not_stale` error in the Architecture guard step, identically at dev d0486978dd and at the round-16 stack top. The step runs in the required derived-artifacts job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each unclassified clock is classified or removed, and the three nodes pass
- [ ] #2 The commit that made them red is named in the notes
<!-- AC:END -->
