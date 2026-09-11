---
id: TASK-32001
title: Implement durable publication journal and crash reconciliation
status: To Do
assignee: []
created_date: '2026-09-07 23:58'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31987
  - task-31988
  - task-32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Interrupted publication is classified using durable journal and actual filesystem evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Interrupted publication is classified using durable journal and actual filesystem evidence.
- [ ] #2 No supported startup opens an ambiguous mixed generation, and original/candidate/rollback evidence is retained.
- [ ] #3 Native crash tests cover every durable transition and refuse unqualified filesystem semantics.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-18)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
