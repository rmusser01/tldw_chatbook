---
id: TASK-38702
title: Keep retained Eval recovery within the existing Windows restore deadline
status: In Progress
created_date: 2026-11-04 22:48
assignee:
- '@codex'
labels:
- bug
- backup-recovery
- windows
- ci
priority: high
documentation:
- Docs/Architecture/backup-recovery.md
- Docs/Architecture/evals.md
updated_date: 2026-11-04 23:03
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3345's current-head Windows Eval recovery lane times out in the complete retained-definitions restore. Restore work advances through staging and installed validation but exceeds the existing 45-second child deadline. Make the affected recovery complete reliably without weakening the deadline, recovery proof, data retention or file authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The original complete-rebackup retained Eval case and affected Eval recovery checks pass on hosted Windows with the restore subprocess deadline still 45 seconds.
- [ ] #2 A measured cause and meaningful regression demonstrate the correction; no timeout increases, skips, retries-to-green, fixture data pruning or validation/durability/ownership bypasses are introduced.
- [ ] #3 Source and candidate removal, inactive retained definitions, rebackup, rejection controls and physical resource retirement retain their original assertions and behavior.
- [ ] #4 Relevant architecture, incident evidence and implementation notes are updated; an independent exact-head review accepts the final change before conditional auto-merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for passive test diagnostics or a mechanical correction preserving all existing recovery boundaries; reassess before an authority, storage or durability contract change.
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md and backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: retain current admission, validation, namespace, source, durability and physical-retirement requirements.
1. Preserve the exact current PR/head/base, failed Windows logs, all-job metadata and original diagnostic artifact. Reproduce the unchanged case locally with the original 45-second restore limit; retain the result as a Mac control only.
2. Measure restore phases and actual call counts. Local profile383 is not Windows proof: Windows uses direct artifact preparation rather than the Mac helper-launch path. Do not infer a Windows cause from local totals.
3. Publish one bounded test-only phase diagnostic through the existing on_staged callback after durable staging/source retirement. Emit only bounded basename/function/line/count/time metadata to the existing captured child log. Keep the original script assertions, restore call, graph and 45-second parent limit. Preserve primary exceptions and observer cleanup. No instrumented success qualifies the final fix.
4. Use actual Windows phase evidence to identify the smallest shared correction. Add a meaningful failing regression and obtain independent review before changing production code. Do not pool helpers, cache validation verdicts, remove barriers, prune data or extend timeouts without an independently justified existing contract.
5. Remove temporary profiling after diagnosis while retaining its source and evidence. Run the original affected cases and connected negative controls uninstrumented; publish a normally committed exact-head reviewed final change while auto-merge is off. Require actual new-head hosted Windows, Perf and required CI.
6. Once genuine gates and DoD pass, complete the finite authorized task reconciliation and independently rebind that metadata head, require its own hosted checks, then enable auto-merge under repository queue rules.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
