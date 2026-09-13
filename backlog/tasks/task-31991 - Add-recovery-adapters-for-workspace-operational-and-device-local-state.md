---
id: TASK-31991
title: Add recovery adapters for workspace operational and device-local state
status: Done
assignee:
  - codex
created_date: '2026-09-07 23:51'
updated_date: '2026-09-08 08:55'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Operational history and recoverable device-local bytes are retained while their execution authority remains quarantined.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Operational history and recoverable device-local bytes are retained while their execution authority remains quarantined.
- [x] #2 File Notes ownership and ordinary export exclusions remain consistent with ADR-021/059/060.
- [x] #3 All operational persistence census rows have capture, schema, relocation, and activation classification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish importable operational declarations and the specified behavioral RED.
2. Map actual workspace, agent history, subscriptions, scheduling, notification/cursor, MCP, File Notes and lasting-sync durable owners against the census and canonical selectors.
3. Add checked owner-local capture/schema/dependency policies; preserve quarantined histories and never replay pending intent.
4. Keep fresh local authority separate from historical references; capture shared/config scopes and disabled durable owners without inventing memory persistence.
5. Qualify real SQLite/file/process behavior and reconcile exact private-SQLite and producer census rows.
6. Run targeted operational tests and named guards, scoped lint/format and self-review; record actual evidence and commit scoped work.

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of ADR-126, preserving ADR-021/059/060 file authority and selective-export exclusions. Independent review approved the task-scoped capture and inspection boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added 21 frozen operational declarations with checked SQLite/raw capture, exact
installed schemas and profile dependencies. All retained execution, binding,
permission, cursor, receipt and managed-membership claims remain inert history.
Actual FileNotesReplica, receipt, Kanban, briefing audio, run-log and optional pet
stores are mapped; no planned storage was invented. Ordinary raw writer/process
lifetimes participate in admission. Exact producer/SQLite inventories and narrow
lazy package exports are updated under the controller's approved scope rulings.

ADR-126 applies, preserving ADR-021/059/060 ownership and selective export exclusions.
115 operational cases passed, followed by 38 affected pet/constructor checks after
one final behavioral RED/GREEN fix. Writer regressions: 222 passed. Prescribed
registry/census and shared-helper checks passed after correcting stale expected
owner counts/Notes IDs and test fixture scope. Scoped lint/format and diff checks
pass. See `.superpowers/sdd/2026-09-07-complete-local-backup-restore/task-8-report.md`
for exact commands, intermediate failures, qualifications, skips and final evidence.

No fresh live identity/activation ledger or coordinated production drain is claimed:
those remain tasks17/20/21 and task10 respectively. Independent review found no Critical or Important findings. Minor duplicate-import,
best-effort admission documentation and baseline warning cleanup remain recorded for
the final branch review; no runtime qualification is inferred from those notes.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-8)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
