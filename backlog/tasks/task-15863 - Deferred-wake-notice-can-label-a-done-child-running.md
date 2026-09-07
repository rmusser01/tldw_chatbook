---
id: TASK-15863
title: Deferred wake notice can label a done child 'running'
status: Done
assignee: []
created_date: '2026-08-13 21:44'
updated_date: '2026-08-14 01:23'
labels:
  - fleet
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-2 Task 7 live: a wake deferred behind a non-empty composer draft (scenario 4) composed its notice with '[a687d2cb…] researcher — running — task: …' for a child whose agent_runs row had been 'done' for a full minute, while delivering that child's complete result in the same notice. The immediate-path wakes (#1-#3) and the post-restart poked delivery all said 'done'. compose_wake_notice takes status from the row dict _rows_for returns, which reads runs_db.get_run fresh — so either the read rode a connection holding a stale WAL snapshot (the held-connection trap from the defer-past-first-paint series), or a stand-in/registry status leaked through on this path. Diagnosis is part of the task; the notice is the honesty surface the supervisor acts on, so a wrong status word is a real defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A wake notice composed at delivery time reports the child's terminal status as of that read, on the immediate, deferred, and mount-claim paths
- [x] #2 The mechanism that produced the stale word is identified and pinned by a test that fails on it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the deferral flow headlessly (real AgentRunsDB file) and audit the compose read path.\n2. Identify the mechanism: compose reads get_run on the app-loop thread's held WAL connection; determine how a non-terminal status word can be read at delivery time; pin with a failing test.\n3. Fix: delivery-time freshness guard in _rows_for (settle-recorded terminal status outranks a non-terminal row read - settle fires strictly after the terminal DB write), or a fresh-read seam; mutation-test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on fix/fleet-wake-ui-residue (commit f0324b036).

Mechanism identified AND demonstrated (AC#2): compose-time get_run rides the app-loop
thread's per-thread held WAL connection, and in Python sqlite3 ANY unfinalized statement
on a connection holds its implicit read transaction open, pinning that connection's
snapshot -- every later read then reports the world as of the pin. The registry can
never produce the word itself: settle-hook and ledger entries are terminal by
construction (the settle hook fires strictly after the terminal DB write; run_child's
finally ordering). Tests/Chat/test_console_fleet_wake_staleness.py holds a pinned
snapshot open, commits the child's terminal write on another thread, proves get_run
still reads 'running', then drives the full deferral+delivery flow: against unfixed
production the notice carried the exact live word ('— running', no result).

Honesty note on the live evidence: the observed pairing "running + complete result"
cannot come from one row read (result and terminal status commit in ONE set_status
transaction), so the live "complete result" detail was most likely the '(no result
recorded...)' line plus supervisor-context content; the pinned-snapshot mechanism
reproduces the header word exactly.

Fix: a pending run's row reading NON-terminal is treated as the provably-stale read it
is and re-read through AgentRunsDB.get_run_fresh (dedicated, immediately-closed
connection -- cannot inherit a pin), recovering both the terminal word and the result;
a handle without the fresh seam falls back to the settle-recorded terminal word.
Read-only additive DB method; stamp/exactly-once semantics untouched. Mutations: 5 run,
5 killed.

Live: the immediate-path and restart/mount-claim-path notices both read '— done'
(panes archived). The deferred live path could not be re-driven end-to-end because the
user-wins-ties probe is blind to a live-typed draft -- a separate regression filed as
task-15970 with pane+instrumentation evidence. Also recorded: the same pin makes
mark_wake_delivered's BEGIN IMMEDIATE fail ('cannot start a transaction within a
transaction') -- logged/tolerated by design (risk = one re-announce), noted for a
future hardening pass.
<!-- SECTION:NOTES:END -->
