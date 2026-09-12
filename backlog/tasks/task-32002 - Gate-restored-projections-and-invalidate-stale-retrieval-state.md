---
id: TASK-32002
title: Gate restored projections and invalidate stale retrieval state
status: To Do
assignee: []
created_date: '2026-09-07 23:58'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31989
  - task-31990
  - task-31991
  - task-32000
  - task-32001
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
- [ ] #2 Omitted/shared indexes obey explicit previewed retirement/quarantine and scope rules.
- [ ] #3 Retrieval resumes only after qualified compatibility and reconciliation, without automatic rebuilds.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-19)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.
