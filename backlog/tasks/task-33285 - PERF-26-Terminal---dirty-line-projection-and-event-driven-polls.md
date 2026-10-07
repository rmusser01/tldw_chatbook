---
id: TASK-33285
title: 'PERF-26: Terminal - dirty-line projection and event-driven polls'
status: To Do
created_date: 2026-09-28 18:03
labels:
- performance
- terminal
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console Terminal re-projects every cell of every session per frame (11-20 ms per session per frame) and at least twice per keystroke. Output parsing re-projects every scrolled line, capping throughput at 0.19 MB/s under the GIL. Each session polls at 200 Hz (runtime bridge) and 100 Hz (input flush) on top of the 50 Hz ownership monitor (TASK-31503). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-26; every issue with file:line is listed under PERF-26 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frame rendering cost is proportional to changed lines
- [ ] #2 Hidden sessions do no per-frame projection
- [ ] #3 Runtime-bridge and input-flush waits are event-driven
- [ ] #4 An idle terminal session costs under 0.2% of a core
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
