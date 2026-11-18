---
id: TASK-41516
title: "The 48 s TASK-41123 node sits in the serial pr-fast-lane"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/Backup_Recovery
dependencies:
- TASK-41123
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by bkreview (round 15). The TASK-41123 node takes 48 s and runs in the serial `pr-fast-lane`, which charges every PR. Moving it to a parallel lane keeps the fast lane fast.

Also: `Tests/Chat/test_fleet_wake_retry.py` is named by no PR lane; TASK-41138 only marked it (there is no `Tests/UI/test_fleet_wake_retry.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The node runs on a lane that is not the serial pr-fast-lane, with before and after fast-lane times recorded
- [ ] #2 The node is still run by at least one PR lane
- [ ] #3 `Tests/Chat/test_fleet_wake_retry.py` runs on a PR lane, with its time recorded
<!-- AC:END -->
