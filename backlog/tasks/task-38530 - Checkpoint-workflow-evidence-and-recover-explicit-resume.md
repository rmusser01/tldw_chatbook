---
id: TASK-38530
title: Checkpoint workflow evidence and recover explicit resume
status: In Progress
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
updated_date: 2026-11-01 21:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 7 of the accepted native shared-worktree workflow design so checkpoint workflow evidence and recover explicit resume. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Required handoffs/checkpoints persist independently of optional logs and remain protected from pruning while active or uncertain.
- [ ] #2 Exact manifests/verifier identities, typed process results and current binding/target evidence govern phase advancement and final review.
- [ ] #3 One startup recovery audit retains uncertain charges and requires explicit Resume; quality review never clears cleanup uncertainty or authorizes replay.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect actual native verifier, required-result, physical ownership and recovery contracts; record controller preflight rulings in task-7-context.md before strict model/schema changes.
2. Add bounded approved VerificationSpec definitions with host-derived exact executable-wrapper command; preserve legacy absent-field canonical bytes. Extend native raw-shell executor/runtime/provider with sanitized bounded required-output capture before diagnostic truncation; scripts remain refused.
3. Preserve original native failure receipts for durable-row launch failures; service requires original authorization/receipt/owner drain, never row status or a boolean alone.
4. Publish native checkpoint, required result, retirement and accounting under the existing FULL transaction and reservation caps; add exact v27 migration/recovery catalog.
5. Implement separate upcoming-phase approval and explicit final quality review, exact freshness, existing one-audit recovery projection and clean explicit Resume without resetting counters or reconstructing execution authority.
6. Run targeted native/file/SQLite, fresh-process boundary, compatibility/archive and named Goal regressions; scoped static checks, preservation, self-review and Task8 handoff report. Leave AC/status pending independent gate.
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Implements accepted typed-evidence/recovery contract using existing native raw execution and storage owners; approved minimum producer extensions and compatibility rulings are recorded in task-7-context.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented native required verifier capture, original launch/failure/drain custody, atomic FULL checkpoint settlement and conservative explicit recovery under ADR-285. Schema v27 adds immutable checkpoint sidecars and inert Resume identities while preserving historical v25/v26 canonical bytes. File inputs remain permission-respecting references; same-phase consumers require fresh native producer handoffs. Completed-phase quality review is separate from upcoming approval and unknown execution closure.

Targeted verification: 189 workflow tests passed; named Goal/raw/schema/backup selection had 225 passed, 1 Windows-only skip and 1 documented profile-source refusal, whose exact supported-profile diagnostic passed. Final evidence/recovery/freshness selection passed 33 tests. Scoped lint has no new diagnostics; unrelated work/index preserved. Full evidence, commands, inherited qualification limits and exact coordinator handoff are in `.superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-7-report.md`. Service Resume remains inert: the later coordinator must freshly prepare actual provider/tools/permissions and new remaining-task phase custody. AC/status deliberately remain pending independent review.

ADR: [ADR-285](../decisions/285-native-phased-agent-workflows.md); narrow producer/storage amendments are documented in the linked spec and agent-runtime architecture reference.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
