---
id: TASK-33286
title: 'PERF-27: Notes sync and Personal Context data paths'
status: To Do
created_date: 2026-09-28 18:04
dependencies:
- TASK-33268
labels:
- performance
- notes
- personal-context
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Notes sync:
- the identity fallback does O(bindings x files) sha256 work on the event loop (2 s at 500/10k)
- observe_root and planning run on the loop
- the organization inventory is O(n^2)
- the replica's FTS delete by UNINDEXED columns scans the whole index per upsert

Personal Context:
- Settings > My Profile load runs without read_operation (346 hardened connects, about 16 s)
- each interview answer costs about 117 connects plus keychain calls (about 5.5 s)
- repository lists are N+1
- ABSENT and DISABLED profiles still pay connects and keychain hits per send (TASK-31504) Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-27; every issue with file:line is listed under PERF-27 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes sync planning and identity matching run off the event loop in O(bindings + files)
- [ ] #2 Replica upserts do not scan the FTS index
- [ ] #3 My Profile load and interview steps use a single read operation (connect counts pinned by test)
- [ ] #4 An ABSENT or DISABLED personal-context profile costs zero connects and zero keychain calls per send
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
