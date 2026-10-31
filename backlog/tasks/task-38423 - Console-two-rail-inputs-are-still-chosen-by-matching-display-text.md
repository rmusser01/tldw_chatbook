---
id: TASK-38423
title: 'Console: two rail inputs are still chosen by matching display text'
status: To Do
assignee: []
created_date: '2026-10-31 03:31'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33621.11 moved the Inspect rail badge onto row codes and statuses. Two inputs to the rail still decide from display text, by equality or prefix rather than substring: the staged badge compares the staged summary against a fixed set of 'nothing staged' sentences (console_rail_state), and the Retrieval row's status is blocked when its source summary starts with 'missing' (console_display_state). A reworded summary silently changes the badge or the row status, the same fragility TASK-33621.11 removed elsewhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The staged badge is chosen from a structured staged count or state, not the summary sentence
- [ ] #2 The Retrieval row's status comes from a structured retrieval state, not a prefix of its summary
- [ ] #3 Tests reword each summary and show the badge and status unchanged
<!-- AC:END -->
