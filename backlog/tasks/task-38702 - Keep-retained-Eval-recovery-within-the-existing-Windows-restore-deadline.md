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
updated_date: 2026-11-04 23:38
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
Actual diagnostic head fc259bed Windows run41930900157/job126927201861 failed before restore: native capture rediscovery refused scope_changed after primary ChaChaNotes became unavailable; seven changed identities included two transient sidecar declarations. No restore phase profile was emitted, so no Windows cost conclusion or production fix is selected. Retained original child log and artifact13370879221. Next bounded diagnostic reuses existing tested observe_capture_review around original public child, preserving every native result/cleanup/assertion and emitting only its existing bounded metadata to captured restart-output.log. This identifies exact snapshot refusal/member fields without workflow changes or an unchanged-head rerun. Original restore45 and all source graph/validation/ownership/durability remain unchanged; instrumented success is not qualification.
Source review identified a concrete queued constructor read: app_service_wiring requests Console attention; its retained task runs AttentionMarksRead via run_preparation_read, opens a native DB transaction, then closes/checkpoints its worker connection. Both shared _PUBLIC and _REOPEN fixtures preview synchronously before their first event-loop yield, allowing this accepted initialization read to overlap capture rediscovery. Before fixing fixture ordering, add an actual-stock-App lifecycle regression requiring successful retained task/native tuple return and no preparation reads/offthread native participant leases before unchanged preview/capture. Observe RED, then await the actual retained task before each shared preview (no sleeps/disabling/cancel/retry); retain native concurrent-snapshot refusal controls. This corrects a proven fixture scheduling hazard, not a claim of exact Windows thread attribution or the unmeasured original restore45 cause. Existing ADR125/126; no new production boundary.
Final fixture regression evidence: corrected native stock read observation uses only off-thread/successful-tuple booleans, preserving all three actual fixture marks. The initial empty-projection/empty-return assumptions were wrong and their failures remain retained (epochs387–390); they are not production regressions or counted passes. Final exact regression391 failed on original pending-task ordering in3.56s; exact reviewed wait restored, epoch392 passed all10 original retained/rejection/native-lifetime/startup controls in74.60s (original retained journey47.22s; regression16.37s). Both runs physically reaped with source_changes[]. Epoch385 also passed95 original diagnostic/observer controls. Ruff check/format and diff checks pass on allthree changed Python files. Independent review accepted fixture/diagnostic correctness after both cleanup findings were fixed; Windows restore45 performance and final head review/CI remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
