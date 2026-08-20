---
id: TASK-19011
title: Cut over atomically from legacy to lasting Notes sync
status: To Do
assignee: []
created_date: '2026-08-20 07:52'
labels:
  - notes
  - sync
  - integration
dependencies:
  - TASK-19000
  - TASK-19003
  - TASK-19006
  - TASK-19007
  - TASK-19008
  - TASK-19009
  - TASK-19010
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ship a restart-boundary cutover with no legacy admission path, migrate incomplete legacy evidence into paused candidates, swap the Notes entry points, remove legacy timers and config writes, and only then allow reviewed local root activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cutover release contains no legacy admission path; after a normal restart it migrates incomplete legacy evidence into paused candidates, swaps toolbar/navigation, records the cutover marker, enables the code-owned cutover admission, and only then permits reviewed local-root activation.
- [ ] #2 No production import, timer, handler, worker group, configuration write, or construction path can activate the legacy engine or service after cutover.
- [ ] #3 The application-owned lasting runtime is the only Notes filesystem mutation owner, and automated source/AST guards prove no reachable dual-owner state.
- [ ] #4 Legacy configuration, note columns, sessions, and conflicts remain read-only migration/history inputs for the compatibility window and are never dual-written or presented as lasting journal state.
- [ ] #5 If the replacement runtime is unavailable, `Keep a folder synced` fails closed with the nearest valid action and never falls back to legacy mutation.
- [ ] #6 Production user documentation and the approved design accurately describe the new entry points, status, attention, recovery, and local-only server gate.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: the accepted ADRs explicitly require the one-way fail-closed cutover and forbid concurrent legacy and lasting owners.
