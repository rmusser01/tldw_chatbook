---
id: TASK-32000
title: Plan and stage both restore destinations with explicit dependency mapping
status: To Do
assignee: []
created_date: '2026-09-07 23:57'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31995
  - task-31996
  - task-31997
  - task-31999
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
- [ ] #2 Replacement preserves unknown data until reviewed and includes managed retirement/rollback scope.
- [ ] #3 Isolated restore works independently of damaged current config and rejects source aliases or untrusted destination authority.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-17)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
