---
id: TASK-3011
title: Workspace_DB opens a fresh private-SQLite connection per query (1,352 in one Console push)
status: Done
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Thread audit: consumers are the main asyncio loop (screen/controller/registry service; Textual workers default to same-thread) plus potentially the agent worker thread — thread-local storage is correct for both by construction.
2. RED: spy `connect_private_sqlite` at the base-db seam — after warm-up, 10 registry reads must open 0 new connections (today: 1 per call); a failing transaction must roll back AND leave the held connection usable; two threads must get two distinct connections.
3. GREEN: follow the established ChaChaNotes idiom (thread-local held connection, liveness check with transparent reopen, PRAGMA foreign_keys at open) — WorkspaceDB's `connection()`/`transaction()` stop `closing()` per use; `close()` closes the current thread's connection.
4. Gate: Workspaces test dir + the A/B push probe from task-2902's notes for the measured win.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] A Console push opens a bounded, small number of Workspace_DB connections (target: 1 per thread), verified by the same profiling method.
- [x] Thread-safety of the reused connection is explicitly verified for every current caller.
- [x] Existing Workspace_DB tests green.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
WorkspaceDB now holds one connection per thread, following ChaChaNotes' `_get_thread_connection` idiom including the task-261 idle-gated liveness ping (a per-call `SELECT 1` would have doubled statement count on exactly this hot path); `connection()`/`transaction()` reuse the held connection, `close()` tears down the current thread's. Thread audit: consumers are the main asyncio loop plus the agent worker thread — thread-local is correct for both by construction, pinned by the two-threads test.

**Measured (interleaved A/B push probes, dev vs branch, same machine): settled push 2.93–3.29s → 1.18–1.29s (~60%)** — far beyond the 0.64s the profile attributed to `connect_private_sqlite` alone, because every post-mount sync pass re-reads workspace state and each read previously paid full connection setup. Tests: reuse pin (spy on the base-db seam: 20 reads, 0 new connections — watched RED at 1-per-read), failed-transaction rollback + connection-stays-usable guard, per-thread isolation guard; full Workspaces dir + Tools workspace-roots + console consumer gate: 244 + 364 passed. Probe-time widget count differs by one between dev (359) and branch (358) — a loading indicator still mounted at dev's slower settle moment, not a DOM change (diff is DB-only). Follow-up task-3012: `AgentRuns_DB` documents itself as following "the Workspace_DB pattern: per-call connections" — same fix applies. Files: tldw_chatbook/DB/Workspace_DB.py, Tests/Workspaces/test_workspace_db_connection_reuse.py.
<!-- SECTION:NOTES:END -->
