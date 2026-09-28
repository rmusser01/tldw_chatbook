---
id: TASK-33267
title: 'PERF-08: Amortize Backup_Recovery storage admission (ADR-126 amendment)'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33260
labels:
- performance
- backup-recovery
- database
- adr
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every outermost guarded call re-derives admission evidence from disk. It walks directory chains from / with one open() per component, reads registry.json about 7 times and takes a flock, under a process-wide initializing section with a 10 ms poll-wait. That is about 245 opens and 4-15 ms per DB transaction on about 12 DB owners, 207k open() calls to reach _ui_ready, about 3,400 open()/s at idle, and 240 admissions per MCP visit. The backup-maintenance monitor probes at 10 Hz forever. Needs owner decision D1: an ADR-126 amendment allowing generation-scoped admission evidence re-checked through held descriptors. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-08; every issue with file:line is listed under PERF-08 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-126 carries an approved amendment describing generation-scoped admission evidence and the preserved invariant
- [ ] #2 A moved, replaced or re-permissioned admitted directory is still refused before dependent I/O (security tests)
- [ ] #3 Per-transaction admission overhead on ChaChaNotes is under 0.5 ms (benchmark), and transactions on unrelated DBs no longer serialize
- [ ] #4 open() calls to reach _ui_ready fall by at least 80% versus the 840ed2ca58 baseline on the same probe
- [ ] #5 The backup-maintenance monitor no longer polls at 10 Hz, and guarded MCP store reads use a warm cache
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
