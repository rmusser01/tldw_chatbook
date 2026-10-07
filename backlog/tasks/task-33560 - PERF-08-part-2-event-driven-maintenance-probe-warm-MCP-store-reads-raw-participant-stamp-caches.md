---
id: TASK-33560
title: >-
  PERF-08 part 2: event-driven maintenance probe, warm MCP store reads,
  raw-participant stamp caches
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-29 20:29'
updated_date: '2026-10-07 01:25'
labels:
  - performance
  - backup-recovery
  - perf-audit-2026-09
dependencies:
  - TASK-33267
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (assessment/amendment of existing reusable-evidence boundary)
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: extend the existing process-local, stamp-validated allowed-evidence reuse to the remaining guarded paths without changing native permission or drain ownership.
1. Trace the actual 1 Hz maintenance probe, MCP/raw/config companion callers and prior owner approval. Measure the current idle/admission cost and pause latency before changing source.
2. Specify the smallest shared reuse path and complete mutation dependencies under ADR-126; document any required amendment before implementation.
3. Add failing warm-path and reuse-versus-full-derivation controls, then implement cache reuse with per-call gates, complete stamps, settle margin, epoch invalidation, exact physical pins and unchanged fallback reasons.
4. Measure a further >=50% reduction in actual idle opens and owner-approved pause latency without changing work, caps or census ceilings. Run targeted security/completeness/cleanup checks and independent review; close only after all criteria are evidenced.

Reviewed concrete reuse design (2026-10-06): independent preflight Ready after explicitly including every consulted historical path-token chain. Use the current hold-owned _Evidence at Backup_Recovery/generation_witnesses._witnesses for its entire positive source-scope/paired-generation derivation; at Admission.pause_requested for parsed registry/groups only; at raw parent pin proof with a fresh independently owned descriptor; and at config companion metadata under its actual retained registry lock. Complete dependencies cover registry/control records/selector/all relevant current and historical roots/activation generation/required.json, with defensive result copies and fresh lease context. Preserve native contention results, selector/member inode/foreign overlap checks, existing counts, epoch/settle and exact full-derivation fallback. History reads retain migration write authority and validated fresh temporary children. Unknown historical/alias inputs remain fresh rather than guessing completeness. Establish paired actual boot/settled-idle baseline before source edits; microbenchmarks alone do not satisfy AC4. Add omitted-history dependency, mutation, publication race, native contention and physical-FD/uncertainty controls before implementation; measured >=50% actual idle reduction and original owner-approved 1Hz pause response remain completion requirements.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
