---
id: TASK-31998
title: Capture a coherent final inventory with optional external content
status: To Do
assignee: []
created_date: '2026-09-07 23:56'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31993
  - task-31994
  - task-31995
  - task-31996
  - task-31997
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: A completed capture reflects the final inventory under maintenance, including valid in-scope growth and coherent DB/asset dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed capture reflects the final inventory under maintenance, including valid in-scope growth and coherent DB/asset dependencies.
- [ ] #2 Changed scope or budget renews preview safely; partial and optional coverage is accurately reported.
- [ ] #3 Ordinary writers resume after verified capture, before encryption or output transfer.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-15)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
