---
id: TASK-33560
title: 'PERF-08 part 2: event-driven maintenance probe, warm MCP store reads, raw-participant
  stamp caches'
status: To Do
created_date: 2026-09-29 20:29
dependencies:
- TASK-33267
labels:
- performance
- backup-recovery
- perf-audit-2026-09
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split out of PERF-08 (TASK-33267). Part 1 reuses confirmed admission evidence in acquire_storage (ADR-126 amendment, 2026-09-29), which with PERF-06 cut open() calls to _ui_ready by 82%. Part 1 leaves AC #5 undone, plus the other guarded paths that still re-derive on every call. (1) The backup-maintenance monitor probes native pause state at 10 Hz for the whole session (runtime_maintenance.py ~700; about 240 opens/s at idle, measured 2026-09-29). (2) Guarded MCP store reads (mcp_source_participants) run the full handshake on every call. (3) raw_participants._scope re-walks its pin chain, re-parses the registry in pause_requested, and re-runs companion_guard's direct _scope on every config operation. Apply the same stamp-validated reuse under the same ADR-126 amendment. A slower monitor changes how quickly a backup sees the app pause, so that part needs the owner's call before landing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The maintenance monitor no longer probes at 10 Hz, and the change to backup pause latency is measured and approved by the owner
- [ ] #2 Warm guarded MCP store reads skip the admission handshake, with the same oracle and completeness tests as acquire_storage
- [ ] #3 raw_participants' pause probe, pin walk and companion_guard scope reuse stamp-validated evidence, and the reuse-vs-derivation oracle still matches under every mutation
- [ ] #4 Idle open() calls per second fall by at least a further 50% on the boot/idle probe
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
