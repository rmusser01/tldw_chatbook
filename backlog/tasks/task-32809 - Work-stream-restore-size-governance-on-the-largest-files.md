---
id: TASK-32809
title: 'Work stream: restore size governance on the largest files'
status: In Progress
assignee: []
created_date: '2026-09-18 16:47'
updated_date: '2026-09-19 19:48'
labels:
  - core-review
  - review-size
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The size ratchets that are supposed to stop the largest files growing are red in 18 rows and absent entirely for several of the biggest modules, including `app.py` at 21,050 lines and the Console controller at 29,048. This stream re-pins what is red and adds rows for what has no budget at all. It does not open any decomposition.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 3 sub-tasks in PR #2744 (fix/core-review-size), the size-governance stream.
<!-- SECTION:NOTES:END -->
