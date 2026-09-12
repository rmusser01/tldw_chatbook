---
id: TASK-32008
title: Expose startup-independent recovery and first-run restore
status: To Do
assignee: []
created_date: '2026-09-08 00:02'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31988
  - task-31995
  - task-32004
  - task-32005
  - task-32006
  - task-32007
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: First-run and damaged-installation users can inspect and restore without normal startup succeeding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run and damaged-installation users can inspect and restore without normal startup succeeding.
- [ ] #2 Minimal recovery reuses qualified services and never bypasses admission, target verification, or activation gates.
- [ ] #3 CLI and UI credentials stay out of process arguments, environment, logs, and persisted requests.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-05-user-workflows.md#task-25)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
