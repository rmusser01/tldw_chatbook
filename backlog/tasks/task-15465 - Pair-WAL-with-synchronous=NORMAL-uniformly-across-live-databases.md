---
id: TASK-15465
title: Pair WAL with synchronous=NORMAL uniformly across live databases
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Probe-verified in the audit (effective PRAGMAs read from created DB files): eight post-2025 databases run at journal_mode=DELETE + synchronous=FULL — Subscriptions (`DB/Subscriptions_DB.py:184-196`), Workspace (`:37-40`), Library_Collections (`:21-24`), RAG_Indexing, search_history, client_notifications, scheduled_tasks (Research/Writing are dead code, see task-15481) — and five WAL databases never set synchronous=NORMAL, so every commit still fsyncs the WAL: ChaChaNotes (`:2767-2769`), Client_Media (`:737-738`), Prompts (`:438-439`), Evals (`:191-192`), AgentRuns (`:73-75`). DELETE mode additionally makes writers exclusive-lock readers; multiple connections contend on the Subscriptions file with sqlite's default 5.0 s busy timeout — a direct candidate for the reported multi-second stalls on slow disks. `DB/Library_Ingest_Jobs_DB.py:57-61` already does it right (WAL + NORMAL with in-line rationale) and is the template.

Stability notes: WAL+NORMAL is the SQLite-documented safe pairing (app-crash-safe; an OS/power crash may lose the last commits — acceptable for these local caches/stores, document per DB); WAL creates -wal/-shm sidecars, so backup/maintenance surfaces that reference DB file paths must be checked (task-899 history); skip the dead DB modules rather than churn them. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every live application database runs WAL + synchronous=NORMAL, verified by a probe reading effective PRAGMAs from created files
- [ ] #2 A write-stall microbenchmark on one write-hot path (e.g. watchlists item upserts or workspace pin/save) recorded before/after
- [ ] #3 Backup/restore and DB-maintenance surfaces handle -wal/-shm sidecars correctly (test or documented check)
- [ ] #4 Each DB carries a one-line rationale comment like the Library_Ingest_Jobs template
<!-- AC:END -->
