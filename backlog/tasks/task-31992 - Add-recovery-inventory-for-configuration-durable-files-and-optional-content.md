---
id: TASK-31992
title: Add recovery inventory for configuration durable files and optional content
status: To Do
assignee: []
created_date: '2026-09-07 23:52'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
- [ ] #2 External folders/models/temporary media/diagnostics remain opt-in without weakening custom app-owned storage coverage.
- [ ] #3 Aliases, unknown durable entries, missing required files, and unsupported metadata produce truthful coverage results.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-9)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
