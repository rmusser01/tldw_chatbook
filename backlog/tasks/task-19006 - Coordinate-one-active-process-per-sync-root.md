---
id: TASK-19006
title: Coordinate one active process per sync root
status: To Do
assignee: []
created_date: '2026-08-20 07:43'
labels:
  - notes
  - sync
  - lifecycle
dependencies:
  - TASK-19004
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give each lasting root one cross-process watcher and mutation owner, expose passive ownership honestly, and integrate fail-closed lease acquisition and release with application lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical root validation rejects overlap with lasting roots, Folder Files roots, application-private paths, symlink roots, and unsupported writable filesystems.
- [ ] #2 An owner-private OS lock grants watcher and mutation authority to exactly one process per root; database lease rows are status only.
- [ ] #3 Processes that cannot acquire authority expose `Passive in this process` and cannot reconcile, watch, or write.
- [ ] #4 Closing admission blocks new work, lets admitted work reach a durable stage, and releases the lease only after settlement.
- [ ] #5 Multi-process tests prove exactly one mutator and prove forced process death releases operating-system ownership.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/029-local-private-data-boundary.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: ADR-059 already requires one cross-process coordinator; the existing `portalocker` dependency is sufficient.
