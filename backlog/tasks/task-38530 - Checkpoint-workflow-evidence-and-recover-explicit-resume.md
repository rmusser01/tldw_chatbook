---
id: TASK-38530
title: Checkpoint workflow evidence and recover explicit resume
status: Done
created_date: 2026-11-01 17:31
dependencies:
- TASK-38524
- TASK-38525
- TASK-38526
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-7
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
assignee:
- '@codex'
updated_date: 2026-11-01 22:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 7 of the accepted native shared-worktree workflow design so checkpoint workflow evidence and recover explicit resume. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Required handoffs/checkpoints persist independently of optional logs and remain protected from pruning while active or uncertain.
- [x] #2 Exact manifests/verifier identities, typed process results and current binding/target evidence govern phase advancement and final review.
- [x] #3 One startup recovery audit retains uncertain charges and requires explicit Resume; quality review never clears cleanup uncertainty or authorizes replay.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect actual native verifier, required-result, physical ownership and recovery contracts; record controller preflight rulings in task-7-context.md before strict model/schema changes.
2. Add bounded approved VerificationSpec definitions with host-derived exact executable-wrapper command; preserve legacy absent-field canonical bytes. Extend native raw-shell executor/runtime/provider with sanitized bounded required-output capture before diagnostic truncation; scripts remain refused.
3. Preserve original native failure receipts for durable-row launch failures; service requires original authorization/receipt/owner drain, never row status or a boolean alone.
4. Publish native checkpoint, required result, retirement and accounting under the existing FULL transaction and reservation caps; add exact v27 migration/recovery catalog.
5. Implement separate upcoming-phase approval and explicit final quality review, exact freshness, existing one-audit recovery projection and clean explicit Resume without resetting counters or reconstructing execution authority.
6. Run targeted native/file/SQLite, fresh-process boundary, compatibility/archive and named Goal regressions; scoped static checks, preservation, self-review and Task8 handoff report. Leave AC/status pending independent gate.
Review fix round 1: release only proven closed no-run payload slots in their original FULL closure transaction, preserving archive/accounting tombstones; add real native completed-but-unverified quality/dependency refusal coverage (I1/I2).
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Implements accepted typed-evidence/recovery contract using existing native raw execution and storage owners; approved minimum producer extensions and compatibility rulings are recorded in task-7-context.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented bounded native workflow evidence/checkpoint service using existing AgentRunsDB, Goal manifests, raw execution and physical owners. Approved executable wrappers bind exact arguments/source/shell/cwd; required sanitized output captures before optional-log truncation and requires real EOF/cleanup. Original native receipts, including failed launches, plus actual owner drain govern one FULL result/evidence/retirement/accounting commit. Literal inputs remain references; same-phase handoffs require fresh authoritative results. Historical v25/v26 JSON bytes/catalogs remain readable through v27 migration. Separate quality/recovery and passive explicit Resume retain unknown charges/protection; the existing single startup audit projects clean quiescence versus uncertain execution.

Independent review found unused no-run payload capacity and a missing native readiness regression. Both passed scoped re-review after a six-path fix: only conclusively unused slots are tombstoned in the closure transaction; actual launched/published/unknown evidence stays protected. Genuine native RUN_DONE without approved process evidence persists missing and blocks quality/dependent approval. Final fix selection51 passed with real SQLite rollback proof; earlier189 workflow,225 regression passes/1Windows skip/1original raw source-profile refusal, exact qualified rerun1 pass are separate overlapping selections. Final prior behavioral33 passed. Scoped lint has no new diagnostics; original index/unrelated recorded hashes preserved.

ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Direct implementation with narrowly documented source-backed producer/evidence/recovery amendments. Architecture/spec/testing lesson updated. Exact API/commands/evidence in .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-7-report.md. Commits b4c93bcb93 and38c05d602c. Task8 genuine async provider/tool revalidation, fresh remaining-phase custody and native coordinator integration remain later; Task9 UI/Task10 live/full native-sidecar profile archive qualification remain pending. Native exact SQLite sidecar archive and full-profile historical-result bundle are separate evidence levels; no full suite/live/Windows claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Required native evidence, atomic checkpoints and conservative one-audit recovery passed independent spec+quality gate after unused-capacity and missing-evidence test fixes. Actual execution remains disabled until the later coordinator/UI/live qualification slices.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
