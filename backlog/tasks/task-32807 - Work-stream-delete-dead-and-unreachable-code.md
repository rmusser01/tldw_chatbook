---
id: TASK-32807
title: 'Work stream: delete dead and unreachable code'
status: To Do
assignee: []
created_date: '2026-09-18 16:47'
labels:
  - core-review
  - review-dead
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roughly 12,000 lines across the package have no production importer: 21 of 66 top-level widgets, a dead RAG pipeline subsystem, eight Event_Handlers modules (one of them not even importable), the deprecated Tools & Settings window, and a 1,427-line dead CSS string. Several are kept alive only by tests that exist to test them. Deleting them shrinks the surface every other stream has to reason about.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
