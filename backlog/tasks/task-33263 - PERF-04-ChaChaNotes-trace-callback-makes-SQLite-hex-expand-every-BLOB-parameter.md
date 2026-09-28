---
id: TASK-33263
title: 'PERF-04: ChaChaNotes trace callback makes SQLite hex-expand every BLOB parameter'
status: To Do
created_date: 2026-09-28 18:02
labels:
- performance
- database
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DB/base_db.py (~738) installs set_trace_callback on every ChaChaNotes connection only to notice BEGIN/COMMIT/ROLLBACK. CPython then renders the expanded SQL, hex-encoding every BLOB parameter, for the statement and for every trigger and FTS sub-step. Inserting a message with a 3 MiB image measured 1,203 ms versus 9 ms for text, and the Console durable-turn commit holds BEGIN IMMEDIATE for that time. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-04; every issue with file:line is listed under PERF-04 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Transaction-boundary detection no longer requires a trace callback that expands bound parameters
- [ ] #2 The semantic-mutation guard keeps its fail-closed behaviour (existing guard tests pass)
- [ ] #3 Inserting a message with a 3 MiB image through CharactersRAGDB takes under 50 ms in a pinned test or benchmark
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
