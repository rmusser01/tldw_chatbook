---
id: TASK-32803
title: 'Work stream: one timestamp helper, and the comparisons that are already wrong'
status: To Do
assignee: []
created_date: '2026-09-18 16:46'
labels:
  - core-review
  - review-time
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Twelve UTC string shapes are produced by 55 helper copies, SQLite's `CURRENT_TIMESTAMP` adds a thirteenth, and 100 sites compare them lexically. Three comparisons are already wrong for users and one was worked around with `julianday()`. Nothing in the repo guards the format a timestamp is written in. This stream introduces one helper, fixes the live defects, and adds the missing guard.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
