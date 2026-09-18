---
id: TASK-32805
title: 'Work stream: provider and streaming correctness'
status: To Do
assignee: []
created_date: '2026-09-18 16:46'
labels:
  - core-review
  - review-wire
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven of eight streaming handlers leak the HTTP response when a consumer stops the stream, two providers drop their usage block so streamed turns record no tokens, and two summarizers can never return a result at all. These are wire-boundary defects with user-visible consequences: leaked connections, missing cost data, and a feature that always errors.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
