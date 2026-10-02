---
id: TASK-33560
title: 'PERF-08 part 2: event-driven maintenance probe, warm MCP store reads, raw-participant
  stamp caches'
status: In Progress
created_date: 2026-09-29 20:29
dependencies:
- TASK-33267
labels:
- performance
- backup-recovery
- perf-audit-2026-09
priority: medium
updated_date: 2026-10-02 15:49
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
- [ ] #5 Requester-approved ordinary reuse preserves full-byte control freshness, retained predecessor/current native identity/security/gate checks, exact live hold/PID/selection ownership, validation outside global bookkeeping and positive retirement, with actual mutation/race oracle and native macOS/Linux/Windows evidence; cold capture/publication behavior remains unchanged.
- [ ] #6 Identical retained reconstructed historical/current probes demonstrate less than 0.5 ms complete ChaChaNotes admission overhead including entry and retirement and at least 80% fewer opens to actual UI readiness; provenance, failed controls and platform-native costs remain explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile dev113's merged evidence reuse and accepted ADR with the separately requester-approved held-evidence/current-byte design; trace actual consumers and record only load-bearing gaps.
2. Retain an identical reviewed probe on immutable historical/current source before changes, including full transaction entry/retirement and actual readiness. Preserve all failed/earlier identities and numeric goals.
3. Implement only remaining ordinary-hold/native/concurrency requirements, then MCP parse reuse and bounded event/native monitor scheduling, with meaningful fail-first mutation/race/error tests and scoped checks.
4. Measure preserved numeric/native/idle targets, independently review scoped immutable changes, publish separately against latest dev and integrate only with exact-head protections and current checks.
ADR required: yes; reconcile the accepted ADR126 and the requester-approved2026-10-01 amendment before production edits. Reuse existing implementations; no new dependency, subsystem or authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-02 continuation of the requester-approved backup workstream. Current dev113e435 merged part1 acquire_storage stamp evidence (PR2919), so remaining implementation will reuse it where it satisfies the approved2026-10-01 design and strengthen only concrete gaps. The requested contract retains fresh current bytes before parsed reuse, actual native Hold/PID/selection ownership, current pathname/predecessor/security/native-gate barriers, Windows native checks, validation outside global bookkeeping, positive retirement and unchanged cold capture/publication routes. Monitor at most1s after the prior native probe settles plus relevant immediate local lifecycle probes was approved by the requester; native/cancellation/maintenance deadlines remain unchanged. Existing TASK-33267 upstream Done record and microbench claims retain their identities. The independently reviewed reconstructed full-transaction/boot probe has separately reviewed signal-status correction6c90; original495/historical receipts are preoptimization controls, not dev113 qualification. Read-only applicability map /private/tmp/task33560-reconciliation-report.md is in progress before production edits. All original numerical and native/isolation gates remain; no full suite, dependency, cache authority, deadline change, blind retry or duplicate original matrix.
Read-only architecture reconciliation completed at exact dev113e435: /private/tmp/task33560-reconciliation-report.md. Reuse existing _Hold/lease/retirement/coordinator/native helpers and actual reuse-vs-full-derivation oracle, not a parallel subsystem. Concrete approved-contract gaps remain: current bounded full-byte comparison before immutable parse reuse, retained root-to-leaf predecessor/native barriers, final generation/selection/pause check after outside-lock warm validation, filesystem work still under the global mutex, and native Windows qualification. The accepted upstream stamp/epoch/settlement policy is preserved as historical part1; it does not lower the separately owner-approved current-byte contract. Existing full derivation legitimately allows some rename/recreate cases; drop stale evidence and reacquire the fully checked current route rather than inventing a new permanent refusal. Task3 consumes this ownership contract for raw/companion/MCP and ≤1s native monitor plus relevant local wakeups. The independently reviewed Task1 I1 signal-status fix6c90 is complete (green15, scoped spec/quality PASS), but its retained34278fac executable is explicitly unmeasured; original495/historical receipts are not currentdev qualification. Current finite followup rebase check at bbe08/dev113:773cases760PASS12worker setup failures(config_source_not_installed),0ERROR1upstream oracle skip(no unbound profile record),0undrained parent network,7616 source pins stable. Scoped producer/consumer lifetime diagnosis ongoing underTASK33370, no production admission changes yet. Origin dev has since advancedeba4305(Console Delete/Undo/DB/UI); assess relevant changes before immutable current timing, do not relabel113.
Tracking reconciliation before production edits: added the separately requester-approved complete-boundary numeric and stricter native/current-byte/concurrency acceptance outcomes here, because upstreamTASK33267 is already Done/part1. These are the existing approved2026-10-01 requirements, not new functionality or a waiver of the four original remaining criteria. Existing stamp wording in the original description records part1's historical policy; remaining reuse follows the linked approved full-byte contract.
Preparation checkpoint: approved documentation/tracking restored from preserved normal Git stashad4f0bd, whose exact patchSHA7994d6ae2908e3cd5d1bf564c5add79f0eeda00457faf4be46d9576e04864dbf is retained under /private/tmp/task33560-metadata-preserved. This preserves all six existing owner-approved outcomes and upstreamTASK33267 Done/part1. PR2955 latestdev fixture/benchmark qualification now passes777 collected(776PASS1inapplicableSKIP),0fail/error, source06165659/dev eba4305, all7622 Python/SQL pins stable. This is test/benchmark followup evidence, not admission performance/native qualification. Scope/Task1/reconciliation metadata will be committed normally with workstream bookkeeping; Task2/3/4 production/performance work remains pending on separate admission branch. The fixed34278fac probe still requires fresh identical historical/current-source measurements before product edits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
