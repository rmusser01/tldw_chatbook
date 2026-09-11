---
id: TASK-31996
title: Validate and migrate imported SQLite under restricted owner policies
status: To Do
assignee: []
created_date: '2026-09-07 23:55'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31989
  - task-31990
  - task-31991
  - task-31994
  - task-31995
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Imported schema is qualified before migration and cannot activate unexpected SQL or application side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Imported schema is qualified before migration and cannot activate unexpected SQL or application side effects.
- [ ] #2 Valid supported SQLite/FTS stores migrate and validate under restricted connections.
- [ ] #3 Unsupported capabilities, schemas, and resource failures remain explicit without unrestricted fallbacks.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-13)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
