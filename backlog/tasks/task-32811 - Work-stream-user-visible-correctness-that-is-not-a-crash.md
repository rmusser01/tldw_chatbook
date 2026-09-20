---
id: TASK-32811
title: 'Work stream: user-visible correctness that is not a crash'
status: To Do
assignee: []
created_date: '2026-09-18 16:47'
labels:
  - core-review
  - review-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Features that silently do nothing or show the wrong thing: a button that can never open a picker, an OS-video fallback that always raises, a diagnostic that reports every optional dependency missing, a picker that shows one template while submitting another, and a metadata pane that is replaced by an entire transcript.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
