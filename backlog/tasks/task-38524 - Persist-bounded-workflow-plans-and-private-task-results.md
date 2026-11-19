---
id: TASK-38524
title: Persist bounded workflow plans and private task results
status: Done
created_date: 2026-11-01 17:30
dependencies:
- TASK-38523
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-1
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
assignee:
- '@codex'
updated_date: 2026-11-01 18:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 1 of the accepted native shared-worktree workflow design so persist bounded workflow plans and private task results. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict plans enforce ordered dependencies, captured scope/targets, byte and count limits without execution authority.
- [x] #2 Workflow creation, revision control and private result reservations are durable, scoped, idempotent and preserve accounting tombstones.
- [x] #3 Real migrations/reopen and exact recovery/backup schema validation preserve existing run/Goal/fleet data and keep restored workflows inert.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add strict contract and real SQLite boundary tests; run focused models/store checks red.
2. Implement passive immutable workflow state and bounded private results in the existing FULL transaction owner.
3. Add AgentRunsDB v24-to-v25 migration, exact recovery catalog/validation/staging and inert backup coverage.
4. Run new focused checks plus named automatic-work/Goal/recovery regressions, scoped lint/format and self-review.
5. Update database architecture and notes, commit only owned paths; leave In Progress for controller review.
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: direct passive persistence implementation of accepted ADR-285; execution and permission owners unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented passive ADR-285 workflow contracts and AgentRunsDB v25 storage: immutable request/plan history, separate control/progress revisions, exact native run/attempt ownership, pre-reserved bounded private results, explicit missing-payload failure and settled-removal accounting/idempotency tombstones. Drain observations are passive metadata supplied by existing native custody; they never release ExecutionOwner or authorize work. No phase execution, admission, coordinator, poller, UI or workflow startup sweep was added.

Actual AgentRuns recovery ownership is DB/recovery_operations.py plus Backup_Recovery/sqlite_validation.py; unrelated recovery_core.py/core_schema.py were correctly left untouched. Added the next actual v24-to-v25 migration, frozen exact catalog and narrow staging allowlist; retained all historical catalogs and kept Goal row validation active at >=24. Historical test fixtures were scoped to their intended target versions, preserving exact constructor/standalone comparisons. Goal annotation-only imports and a new workflow-store identity import were made constructor-free after a fresh private-process RED exposed a self-held restore startup lease; actual Goal termination normalization remains covered. ADR required: no new ADR; direct implementation of backlog/decisions/285-native-phased-agent-workflows.md.

Verification: 137 targeted checks passed in 57.40s (all new modules, all four named migration/recovery regressions, existing Goal models/store/attempts). Post-format/import annotation checks: 38 passed in 0.93s. Real complete capture -> isolated restore -> selected-profile reopen -> complete rebackup passed with unchanged native watchdog/child defaults; archived workflow rows match exactly and both archived copies pass foreign_key_check. New/affected contract modules pass scoped ruff lint/format; shared legacy files have no introduced per-file lint findings against task base 13ded554. Detailed commands, RED/GREEN receipts, catalog/archive proof and limitations are in .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-1-report.md.

Updated Docs/Architecture/database-layer.md for passive storage only and recorded the demonstrated recovery-import trap in lessons-testing-evidence.md. Independent review is recorded below; no full suite or product phase execution was run.

Independent spec/quality gate passed after fix c3b8a44115: attempt-based mandatory evidence validation now rejects missing result/reservation pairs and invalid active/undrained tombstones. Fix RED16 failed/4 passed; final covering run46 passed,1 deselected; lint/format pass. Initial cfde251667 and fix c3b8a44115. Existing cleanup warnings/original unmarked prerequisite failures remain recorded. No full suite. Trusted drain provenance and live execution remain Tasks7–8 of ADR-285.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Passive workflow storage/results implemented and independently reviewed. ADR-285 applies; no live workflow capability is enabled by this slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
