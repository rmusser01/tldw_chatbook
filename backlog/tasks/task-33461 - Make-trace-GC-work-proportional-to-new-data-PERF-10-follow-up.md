---
id: TASK-33461
title: Make trace GC work proportional to new data (PERF-10 follow-up)
status: To Do
created_date: 2026-09-29 19:04
labels:
- performance
- database
- perf-audit-2026-09
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split out of PERF-10 (TASK-33269), which delivered its other acceptance criteria: legacy trace maintenance now parks once normalization is complete and wakes on a new exchange write or a due GC pass, and the completion check no longer takes a write transaction. The physical trace GC pass itself still runs at TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS and does work in proportion to the whole trace ledger rather than to rows written since the previous pass (2026-09-27 structural perf audit, qa/perf-structural-audit-2026-09-27/report.md, PERF-10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A trace GC pass with no rows written since the previous pass does no ledger-wide scan (measured by statement count or query plan, not timing)
- [ ] #2 GC work after N new exchanges is bounded by N, not by total ledger size, with a test that grows the ledger and asserts the per-pass work stays flat
- [ ] #3 Existing trace retention and GC correctness tests stay green
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
