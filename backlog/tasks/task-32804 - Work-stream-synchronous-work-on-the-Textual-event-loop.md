---
id: TASK-32804
title: 'Work stream: synchronous work on the Textual event loop'
status: To Do
assignee: []
created_date: '2026-09-18 16:46'
labels:
  - core-review
  - review-loop
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`get_cli_setting` costs about 11 ms per call even when warm, because the cached read sits behind a per-call storage-admission handshake. Several surfaces call it per keystroke, per row or per timer tick, and others run synchronous sqlite on the loop. The root cost is one task; the rest are per-surface fixes that stop paying it.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
