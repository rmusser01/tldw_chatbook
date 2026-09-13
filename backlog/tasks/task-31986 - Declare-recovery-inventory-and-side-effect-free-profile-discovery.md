---
id: TASK-31986
title: Declare recovery inventory and side-effect-free profile discovery
status: To Do
assignee: []
created_date: '2026-09-07 23:48'
labels:
  - backup-recovery
dependencies:
  - task-31978
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Discovery identifies selected profiles, custom app-owned storage, shared aliases, and durable unknowns without opening services or modifying data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Discovery identifies selected profiles, custom app-owned storage, shared aliases, and durable unknowns without opening services or modifying data.
- [ ] #2 Coverage states and dependency failures accurately distinguish complete, partial, unavailable, excluded, and intentional deletion.
- [ ] #3 Every existing persistence producer has an explicit owner-inventory row, and new unclassified producers fail an architecture guard.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-3)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
