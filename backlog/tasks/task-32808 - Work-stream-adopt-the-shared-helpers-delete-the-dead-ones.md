---
id: TASK-32808
title: 'Work stream: adopt the shared helpers, delete the dead ones'
status: In Progress
assignee: []
created_date: '2026-09-18 16:47'
updated_date: '2026-09-19 21:02'
labels:
  - core-review
  - review-helpers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
453 public helper symbols exist and 207 have zero in-package importers, while the same functionality is hand-rolled nearby: 52 inline truncations against a helper with no callers, 16 byte-size formatters, 7 filename sanitizers with different safety envelopes, 21 bool coercers that disagree about the integer 1. Each task here either adopts one helper everywhere or deletes it.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #2745 opened with the two concrete correctness bugs (.4 AC#1 None-default, .1 decimal-threshold + public helper). The stream's broad consolidations (byte-size 16 sites, bool 23 sites, truncation 52 sites, etc.) remain as focused follow-ups.
<!-- SECTION:NOTES:END -->
