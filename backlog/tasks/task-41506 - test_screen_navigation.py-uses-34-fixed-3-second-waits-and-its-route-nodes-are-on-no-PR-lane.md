---
id: TASK-41506
title: "test_screen_navigation.py uses 34 fixed 3-second waits and its route nodes are on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- navigation
priority: medium
references:
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15). The file holds 34 fixed 3 s waits plus `_wait_for_initial_screen`; at load 85 its 12-node slice took 96.1 s serial, and its route nodes were left off lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fixed waits are replaced by condition waits with a bounded deadline
- [ ] #2 The route nodes run on a PR lane, with their measured serial time recorded
- [ ] #3 The file's wall-clock time at load under 40 falls measurably, with before and after in the notes
<!-- AC:END -->
