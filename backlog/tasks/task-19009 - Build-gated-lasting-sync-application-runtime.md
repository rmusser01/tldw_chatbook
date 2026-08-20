---
id: TASK-19009
title: Build gated lasting-sync application runtime
status: To Do
assignee: []
created_date: '2026-08-20 07:47'
labels:
  - notes
  - sync
  - lifecycle
dependencies:
  - TASK-19005
  - TASK-19006
  - TASK-19007
  - TASK-19008
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the application-owned lasting-sync runtime and hint-only watcher, but keep every lease, reconciliation, watcher, and activation path inert until both the code-owned cutover admission and private cutover marker exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One application-owned runtime starts independently of the Library screen but remains inert until both the code-owned cutover admission and private cutover marker exist.
- [ ] #2 The dependency-free watcher emits root IDs only; events are debounced scheduling hints and never scan, plan, execute, or mutate.
- [ ] #3 After cutover authorization, manual Sync now performs a fresh reviewed check, while automatic work executes only direction-authorized one-sided operations and records durable outcomes.
- [ ] #4 Paused, Offline, Passive, Needs attention, Partial, Failed, and unsupported roots cannot silently resume mutation and always expose a next action.
- [ ] #5 Shutdown closes admission, stops hints, settles or journals the current stage, releases leases, and finishes before generic database/Textual teardown.
- [ ] #6 Production lifecycle tests prove one runtime identity, no Library-screen lifetime ownership, and no lease/watcher/reconciliation/activation when either cutover gate is absent; no new watcher dependency is added.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: ADR-059/073 already define application ownership, hint-only watchers, complete reconciliation, durable status, and shutdown order; a standard-library polling backend avoids a new dependency decision.
