---
id: TASK-31993
title: Integrate maintenance participants across persistence owners
status: To Do
assignee: []
created_date: '2026-09-07 23:53'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31988
  - task-31989
  - task-31990
  - task-31991
  - task-31992
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Every participating persistence owner drains safely and is covered by the shared admission protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every participating persistence owner drains safely and is covered by the shared admission protocol.
- [ ] #2 Unsaved drafts and unfinished cross-store work are neither discarded nor falsely reported captured.
- [ ] #3 Real multi-process evidence proves coherent ownership boundaries and safe resumption without deadlocks.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-10)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
