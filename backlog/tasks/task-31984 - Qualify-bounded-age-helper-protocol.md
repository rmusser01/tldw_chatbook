---
id: TASK-31984
title: Qualify bounded age helper protocol
status: To Do
assignee: []
created_date: '2026-09-07 23:47'
labels:
  - backup-recovery
dependencies:
  - task-31978
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Streaming encrypted round trips interoperate with official age and reject incomplete authentication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streaming encrypted round trips interoperate with official age and reject incomplete authentication.
- [ ] #2 Header, KDF, memory, password transport, cancellation, and child cleanup limits are demonstrated with synthetic data.
- [ ] #3 Unqualified or absent helpers report unavailable before password collection or maintenance; no plaintext fallback occurs.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-01-encryption.md#task-1)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
