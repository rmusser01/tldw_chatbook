---
id: TASK-33287
title: 'PERF-28: DB query hygiene - N+1s, unbounded scans, planner traps'
status: To Do
created_date: 2026-09-28 18:04
dependencies:
- TASK-33268
labels:
- performance
- database
- perf-audit-2026-09
priority: low
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
These remain after the keystone fixes:
- N+1 listings (review sets 2N+1 per TASK-31508, Personal Context lists, scheduling results)
- unbounded fetchall on growing tables
- a startup trace-recovery scan of the calls table
- a full agent_runs scan by launch-wake discovery on every boot
- the media sync_log, which is never pruned and logs full content twice per ingest
- about 10 low-selectivity single-column indexes (on deleted/enabled) that the statistics-free planner can choose over better ones Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-28; every issue with file:line is listed under PERF-28 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each P2 query item in the appendix is fixed or closed with an EXPLAIN QUERY PLAN captured without sqlite_stat1
- [ ] #2 Any new or changed index has a plan pin per the CLAUDE.md index rule
- [ ] #3 media sync_log has a retention bound
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
