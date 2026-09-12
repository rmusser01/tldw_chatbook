---
id: TASK-31987
title: Implement stable maintenance admission and native storage qualification
status: To Do
assignee: []
created_date: '2026-09-07 23:49'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
- [ ] #2 Deadlock, timeout, cancellation, and crashed-holder cases preserve data and durable recovery evidence.
- [ ] #3 Native publication cannot overwrite an existing artifact, and unsupported storage returns an explicit unavailable capability.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-4)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
