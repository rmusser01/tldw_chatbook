---
id: TASK-32804
title: 'Work stream: synchronous work on the Textual event loop'
status: In Progress
assignee: []
created_date: '2026-09-18 16:46'
updated_date: '2026-09-19 23:23'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #2739. Done: .1 (config fastpath), .4 (review-set snapshot memoize), .5 (folder-import non-copy), .8 splash (.1-section read), .9 (semantic query offload + attach memoize), .10 (prompt-search subquery + reconcile batch-hydrate off-compose). .3/.8-Speech: cost resolved by .1, structural ACs are follow-ups. Remaining: .2 (RAG settings 24-handler re-derive), .6 (Console per-row/tick sqlite), .7 (MCP permission/workbench), .11 (Library widget small fixes), .12 (P2 grab-bag).
<!-- SECTION:NOTES:END -->
