---
id: TASK-31994
title: Persist recovered media references and deletion tombstones
status: To Do
assignee: []
created_date: '2026-09-07 23:53'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
  - task-31992
  - task-31993
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Included temporary media becomes durable and resolves correctly across restart and subsequent backups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Included temporary media becomes durable and resolves correctly across restart and subsequent backups.
- [ ] #2 Intentional deletion survives as a tombstone without partial-backup status; unexpected missing required bytes still block completeness.
- [ ] #3 Shared references, recovery holds, explicit cleanup, and interrupted publication/deletion preserve recoverable state.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-11)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
