---
id: TASK-33269
title: 'PERF-10: Legacy trace maintenance - park when complete, incremental trace
  GC'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33268
labels:
- performance
- console
- database
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
New evidence for TASK-31501. The ~1 Hz forever legacy trace-maintenance tick now costs 6-9 ms of CPU per tick, and on an unconnected executor thread it spawns a private-SQLite helper every tick (79 ms wall, about 45 ms child CPU). Trace GC re-marks the whole reachable ledger inside BEGIN IMMEDIATE every 60 s after any send. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-10; every issue with file:line is listed under PERF-10 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Once legacy normalization is logically complete, the maintenance loop performs no periodic DB work until a new exchange is written
- [ ] #2 The completion check does not take a write transaction
- [ ] #3 Trace GC work is proportional to new data, not total ledger size
- [ ] #4 The idle probe shows zero helper spawns and zero write transactions from trace maintenance
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
