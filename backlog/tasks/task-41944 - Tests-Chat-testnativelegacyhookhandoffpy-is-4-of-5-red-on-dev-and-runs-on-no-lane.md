---
id: TASK-41944
title: "Tests/Chat/test_native_legacy_hook_handoff.py is 4 of 5 red on dev and runs on no lane"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- agents
- hooks
priority: low
references:
- Tests/Chat/test_native_legacy_hook_handoff.py
dependencies:
- TASK-41639
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** On dev d0486978dd, 4 of 5 nodes fail ("failed host publication retired the event"; the required-commit node). No CI lane runs the file, so nothing flags it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each red node has a recorded cause (drift with its commit, or a product defect fixed red-first)
- [ ] #2 The file passes and runs on a PR lane
<!-- AC:END -->
