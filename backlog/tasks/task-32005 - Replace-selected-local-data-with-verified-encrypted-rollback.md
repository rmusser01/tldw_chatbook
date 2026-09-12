---
id: TASK-32005
title: Replace selected local data with verified encrypted rollback
status: To Do
assignee: []
created_date: '2026-09-08 00:00'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31993
  - task-31997
  - task-31999
  - task-32000
  - task-32001
  - task-32002
  - task-32003
  - task-32004
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
- [ ] #2 Maintenance spans the entire safety-copy/publication/validation interval and final active inventory matches the approved generation.
- [ ] #3 Interrupted or failed replacement retains recovery evidence and never boots ambiguous or automatically active state.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-22)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
