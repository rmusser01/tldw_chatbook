---
id: TASK-3012
title: AgentRuns_DB opens per-call connections — apply the task-3011 held-connection fix
status: Done
assignee: []
created_date: '2026-08-07 06:00'
labels:
  - db
  - agents
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while gating task-3011: `AgentRuns_DB`'s module docstring says it "follows the Workspace_DB pattern: BaseDB, per-call connections" — the exact anti-pattern task-3011 just removed from WorkspaceDB (which measured ~60% of the Console push). AgentRuns is hot during agent runs (per-step persistence), so each step currently pays full private-SQLite connection setup. Apply the same thread-local held-connection idiom (idle-gated liveness ping, per-thread isolation, close() teardown) with the same three-test shape (reuse pin watched RED, rollback+usable guard, per-thread guard). Note the agent service runs on a worker thread — the thread-local design covers it, but the audit should confirm no connection is shared across the service/main boundary.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Differences from 3011 that must survive: `BEGIN IMMEDIATE` transactions (concurrent primary + sub-agent writers), WAL + busy_timeout pragmas (documented cross-process contention), and the reconcile-on-init sweep. All are per-connection properties — a thread-local held connection keeps every one.
2. Thread audit: agent service worker thread + sub-agent threads + main-loop UI reads — thread-local gives each its own connection; short-lived run threads leak-close at GC exactly as ChaChaNotes' pattern does.
3. RED: the 3011 three-test shape against AgentRunsDB (reuse pin via the base-db spy watched RED; failed-BEGIN-IMMEDIATE rollback + connection-stays-usable; two-thread isolation).
4. GREEN: same thread-local + idle-gated liveness idiom; `close()` teardown.
5. Gate: Agents test files using AgentRunsDB + collect-only.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] Repeated AgentRuns reads/writes on one thread open no new connections after warm-up (spy-pinned, watched RED first).
- [x] Transaction rollback semantics and post-failure usability preserved; per-thread isolation pinned.
- [x] Existing AgentRuns/Agents test files green.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Applied the 3011 idiom (thread-local held connection, idle-gated liveness ping, `close()` teardown) with AgentRuns' specifics preserved: `BEGIN IMMEDIATE` transactions, WAL + busy_timeout pragmas, and reconcile-on-init — all per-connection properties re-applied on every (re)open. One real discovery beyond the mechanical port: **the held connection required `isolation_level = None`** — Python's default mode auto-BEGINs on any DML, and the accumulated implicit transaction made the explicit `BEGIN IMMEDIATE` raise "cannot start a transaction within a transaction". The per-call shape had masked this — and worse, it silently ROLLED BACK any bare DML issued through `connection()` (close-without-commit). Audited every `connection()` site before switching: all read-only except `_initialize_schema`, whose `executescript` self-commits under either mode, so autocommit changes no existing behavior. Tests: the 3011 three-test shape (reuse pin watched RED at one-connection-per-read; BEGIN-IMMEDIATE rollback + connection-stays-usable; two-thread isolation). Gate: full `Tests/Agents/` + change-review screen + agent rail + agent bridge — 938 passed. Files: tldw_chatbook/DB/AgentRuns_DB.py, Tests/Agents/test_agent_runs_db_connection_reuse.py.
<!-- SECTION:NOTES:END -->
