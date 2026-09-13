---
id: TASK-31997
title: Implement staged credential exclusion and encrypted credential recovery
status: To Do
assignee: []
created_date: '2026-09-07 23:56'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31989
  - task-31990
  - task-31991
  - task-31992
  - task-31996
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Default archives remove managed secrets from all supported staged locations, including SQLite remnants, without modifying sources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Default archives remove managed secrets from all supported staged locations, including SQLite remnants, without modifying sources.
- [ ] #2 Credential inclusion and exact rollback require encryption and retain supported values with honest omission reporting.
- [ ] #3 Restoration isolates credential scopes and never overwrites shared entries or claims remote authentication was recovered.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-14)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
