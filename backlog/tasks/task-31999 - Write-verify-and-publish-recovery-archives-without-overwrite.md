---
id: TASK-31999
title: Write verify and publish recovery archives without overwrite
status: To Do
assignee: []
created_date: '2026-09-07 23:57'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31985
  - task-31995
  - task-31998
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Archives created by the writer pass the same bounded reader and carry truthful coverage, directory, and credential metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Archives created by the writer pass the same bounded reader and carry truthful coverage, directory, and credential metadata.
- [ ] #2 Existing backups/source/control files remain unchanged under races and aliases.
- [ ] #3 Failures and cancellation never expose incomplete output as a verified archive.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-16)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
