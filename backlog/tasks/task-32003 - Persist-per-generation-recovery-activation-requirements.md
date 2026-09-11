---
id: TASK-32003
title: Persist per-generation recovery activation requirements
status: To Do
assignee: []
created_date: '2026-09-07 23:59'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31988
  - task-31991
  - task-31992
  - task-32001
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Restored capabilities remain inactive across every supported launch until their own owner review completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restored capabilities remain inactive across every supported launch until their own owner review completes.
- [ ] #2 Missing/corrupt activation records and imported approvals cannot grant execution authority.
- [ ] #3 Safe local inspection works and one owner approval never activates unrelated automation or queued work.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-20)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
