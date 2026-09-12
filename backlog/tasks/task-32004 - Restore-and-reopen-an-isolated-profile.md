---
id: TASK-32004
title: Restore and reopen an isolated profile
status: To Do
assignee: []
created_date: '2026-09-08 00:00'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-32000
  - task-32001
  - task-32002
  - task-32003
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Isolated recovery creates and reopens a separate profile without altering original local data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Isolated recovery creates and reopens a separate profile without altering original local data.
- [ ] #2 Damaged current configuration and databases do not prevent archive-only recovery.
- [ ] #3 Fresh launch respects relocated paths, credential/device isolation, durable activation, and projection readiness.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-21)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
