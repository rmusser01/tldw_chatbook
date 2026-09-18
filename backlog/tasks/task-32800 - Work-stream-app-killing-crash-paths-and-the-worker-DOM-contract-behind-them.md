---
id: TASK-32800
title: 'Work stream: app-killing crash paths and the worker/DOM contract behind them'
status: To Do
assignee: []
created_date: '2026-09-18 16:46'
labels:
  - core-review
  - review-crash
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four findings take the whole app down from an ordinary button, key press or dialog, and three of them violate one contract: a Textual worker must be a coroutine or declare `thread=True`, and an `await` must not resume into a subtree that may have been removed. `exit_on_error` defaults to True, so each is a full app exit rather than a degraded surface. This stream fixes the three crashes and adds the guard that would have caught all of them.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
