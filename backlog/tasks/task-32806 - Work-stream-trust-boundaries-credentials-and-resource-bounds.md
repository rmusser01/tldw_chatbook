---
id: TASK-32806
title: 'Work stream: trust boundaries, credentials and resource bounds'
status: To Do
assignee: []
created_date: '2026-09-18 16:47'
labels:
  - core-review
  - review-security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Credentials that fail the shared validity rule are handed to providers, a transient read error resets the whole MCP permission file to permissive defaults, an always-on tool evaluates unbounded model-supplied arithmetic, and several paths read or write user files with no size cap or process-group control. These are the findings where the consequence is a security or resource boundary rather than a wrong pixel.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
