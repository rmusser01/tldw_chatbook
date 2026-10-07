---
id: TASK-33283
title: 'PERF-24: Watchlists and Schedules - incremental panes, DB off the event loop'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33268
labels:
- performance
- watchlists
- scheduling
- ui
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Schedules:
- moving the cursor onto a reminder runs 3 synchronous ScheduledTasksDB reads on the loop (132 ms)
- each visit runs about 145 ms in on_mount before first paint, plus about 385 ms per visit
- SchedulingService and SyncEngine hide synchronous sqlite behind async def
- mark-all-read is N+1

Watchlists:
- the left-rail tree is remounted on every count refresh
- inspector and content panes recompose on every selection, including j/k navigation Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-24; every issue with file:line is listed under PERF-24 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Schedules interaction performs synchronous sqlite on the event loop
- [ ] #2 SchedulingService async methods do their DB work off the loop
- [ ] #3 Mark-all-read runs as a batched write
- [ ] #4 Watchlists tree and panes update in place on count refresh and selection changes
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
