---
id: TASK-3011
title: Workspace_DB opens a fresh private-SQLite connection per query (1,352 in one Console push)
status: To Do
assignee: []
created_date: '2026-08-07 23:30'
labels:
  - db
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
cProfile of a single ChatScreen push (task-2902 round 2): `Workspace_DB.connection`/`_get_connection` → `connect_private_sqlite` ran **1,352 times — a brand-new SQLite connection per query — costing 0.64s** of the ~2.5s first paint. Every other DB in the app holds its connection; Workspace_DB's per-call `connect_private_sqlite` makes each workspace read pay full connection setup (open, pragmas, private-mode setup).

Fix shape: hold one connection per Workspace_DB instance (thread-affine like the other `base_db` users), or at minimum a connection cache keyed by thread. Audit callers for cross-thread use before switching (the scheduler and screen both read workspace state).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] A Console push opens a bounded, small number of Workspace_DB connections (target: 1 per thread), verified by the same profiling method.
- [ ] Thread-safety of the reused connection is explicitly verified for every current caller.
- [ ] Existing Workspace_DB tests green.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
