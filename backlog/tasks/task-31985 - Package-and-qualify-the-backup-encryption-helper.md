---
id: TASK-31985
title: Package and qualify the backup encryption helper
status: To Do
assignee: []
created_date: '2026-09-07 23:48'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31984
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Qualified wheels work without Go, runtime downloads, or PATH helper substitution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Qualified wheels work without Go, runtime downloads, or PATH helper substitution.
- [ ] #2 Source/editable installation and unsupported-platform behavior are explicit and tested.
- [ ] #3 Native platform evidence, pinned dependencies, integrity/version checks, and upgrade interoperability accompany each advertised tuple.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-01-encryption.md#task-2)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
