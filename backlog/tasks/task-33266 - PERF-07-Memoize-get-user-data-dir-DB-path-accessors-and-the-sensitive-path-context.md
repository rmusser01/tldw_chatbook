---
id: TASK-33266
title: 'PERF-07: Memoize get_user_data_dir, DB-path accessors and the sensitive-path
  context'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33260
labels:
- performance
- config
- security
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
get_user_data_dir() (config.py ~9433; 203 call sites, 29 on the boot path) is uncached. It runs 5 admission scopes, the data-root file lock and a secure_private_directory walk: 29-53 ms and 1,000-1,700 open() calls per call. resolve_sensitive_context calls it 19-20 times (about 607 ms per resolution). That runs for every agent file/git/patch tool call, every @-reference, twice in RunLogWriter.bind per send, for the emergency-stop path on every send and on every 30 s scheduler tick, and for every RAGConfig(). Needs owner decision D2: a generation-keyed memo with a leaf identity re-check instead of a per-call chain walk. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-07; every issue with file:line is listed under PERF-07 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Owner decision D2 is recorded (ADR amendment or task note) before merge
- [ ] #2 get_user_data_dir and the get_*_db_path accessors resolve at most once per config generation and data-dir setting, with a documented identity re-check
- [ ] #3 resolve_sensitive_context and default_emergency_stop_path are memoized on the same key
- [ ] #4 Replacing or re-permissioning the data directory is still detected (security test)
- [ ] #5 Main-thread get_user_data_dir calls before _ui_ready drop from about 40 to at most 2
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
