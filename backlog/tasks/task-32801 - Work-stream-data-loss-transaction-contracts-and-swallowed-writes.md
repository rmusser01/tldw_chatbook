---
id: TASK-32801
title: 'Work stream: data loss, transaction contracts and swallowed writes'
status: To Do
assignee: []
created_date: '2026-09-18 16:46'
labels:
  - core-review
  - review-data
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The store template lets a write method assume an outer transaction. Two stores have a method that does DML on a held legacy-isolation connection anyway; in Media a shipped caller opens no transaction, so every later write on that connection is rolled back at close. Around it sit swallowed write failures, a connection leak and a lock-order inversion. This stream closes the transaction contract and makes failed writes audible.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
