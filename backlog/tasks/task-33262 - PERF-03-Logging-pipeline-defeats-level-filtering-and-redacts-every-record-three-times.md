---
id: TASK-33262
title: 'PERF-03: Logging pipeline defeats level filtering and redacts every record
  three times'
status: To Do
created_date: 2026-09-28 18:02
labels:
- performance
- logging
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Logging_Config.py forwards every loguru record to stdlib at level TRACE, so loguru's early level check never fires. Every logger.debug costs about 7-8 us instead of 0.15 us, opt(lazy=True) guards (TASK-275) are defeated, and a dropped opt(exception=True).debug formats a full traceback. Every INFO+ record is redacted three times (shouldRollover, emit, Logs buffer), about 340 us, and flushed synchronously on the emitting thread, the event loop included. The DB layer and importer log INFO per row. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-03; every issue with file:line is listed under PERF-03 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A dropped debug record costs under 1 us (measured) and opt(lazy=True) callables are not evaluated when DEBUG is off
- [ ] #2 Each INFO+ record is redacted exactly once, with redaction regression tests unchanged or extended
- [ ] #3 File and Logs-buffer handlers do not perform file I/O on the event-loop thread
- [ ] #4 Per-row DB/importer INFO logs are demoted, and app-owned worker transitions no longer emit a WARNING each
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
