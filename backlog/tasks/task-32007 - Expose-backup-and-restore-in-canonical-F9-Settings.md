---
id: TASK-32007
title: Expose backup and restore in canonical F9 Settings
status: To Do
assignee: []
created_date: '2026-09-08 00:01'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31999
  - task-32004
  - task-32005
  - task-32006
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Users can discover and execute create/inspect/both restore modes from F9 with truthful coverage, risks, and progress.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can discover and execute create/inspect/both restore modes from F9 with truthful coverage, risks, and progress.
- [ ] #2 Recovery copies and isolated profiles are inspectable and actionable through the canonical UI.
- [ ] #3 Product-level mounted/live evidence verifies actual services and keyboard/navigation behavior without unintended execution.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-05-user-workflows.md#task-24)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
