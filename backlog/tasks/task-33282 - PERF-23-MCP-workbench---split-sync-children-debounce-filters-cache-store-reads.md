---
id: TASK-33282
title: 'PERF-23: MCP workbench - split _sync_children, debounce filters, cache store
  reads'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33267
labels:
- performance
- mcp
- ui
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The monolithic MCP workbench _sync_children re-reads the stores, re-derives the whole catalog and rebuilds all five DataTables on every interaction (TASK-32804.7). It runs twice per warm visit. Arrow keys do guarded store reads plus an inspector remount. Space in the permissions matrix costs about 100 ms. Filter inputs rebuild the DataTable per keystroke with no debounce. A 4 Hz save-status poll forces a full relayout 4 times a second. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-23; every issue with file:line is listed under PERF-23 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An interaction rebuilds only the affected table/inspector
- [ ] #2 A warm visit runs a single sync pass
- [ ] #3 Filter inputs are debounced
- [ ] #4 The save-status indicator updates only when its text changes
- [ ] #5 Arrow-key navigation performs no store reads on the event loop
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
