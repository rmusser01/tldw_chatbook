---
id: TASK-32802
title: 'Work stream: user text silently deleted or mangled by markup escaping'
status: To Do
assignee: []
created_date: '2026-09-18 16:46'
labels:
  - core-review
  - review-markup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`rich.markup.escape` does not protect Textual 8: its pattern only escapes tags starting `[a-z#/@`, so a title like `[TODO] Q3 plan` loses its tag and `[IMPORTANT]` renders as an empty row, across roughly 40 sites. Two more sites escape in the other direction, into markup-off surfaces, so `R&D Report` reaches the user as `R&amp;D Report`. The repo already contains the correct escaper and applies it to exactly one surface; every pinning test uses a lowercase tag, the one case the broken escape handles.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->
