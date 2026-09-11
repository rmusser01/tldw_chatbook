---
id: TASK-31995
title: Implement bounded immutable archive inspection
status: To Do
assignee: []
created_date: '2026-09-07 23:54'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31984
  - task-31985
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Inspection uses completed immutable input bound to a digest and validates manifest/payload integrity under bounded resource use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inspection uses completed immutable input bound to a digest and validates manifest/payload integrity under bounded resource use.
- [ ] #2 Malformed, ambiguous, hostile, or unsupported archives cannot select destination paths or execute content.
- [ ] #3 Limits apply during source copy and decryption before untrusted manifest information is available.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-12)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
